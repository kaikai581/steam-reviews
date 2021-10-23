#!/usr/bin/env python
'''
This script is based on the website.
https://towardsdatascience.com/sentiment-analysis-using-lstm-and-glove-embeddings-99223a87fe8e
'''

from keras.layers import LSTM, Dropout, Dense, Input
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

import keras
import numpy as np
import os
import pandas as pd
import re
import string

stopwords = set([ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any",  "are", "as", "at", "be", "because", 
             "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during",
             "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", 
             "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into",
             "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or",
             "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", 
             "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's",
             "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up",
             "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's",
             "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've",
             "your", "yours", "yourself", "yourselves" ])

def clean_review(data):
    data_without_stopwords = remove_stopwords(data)
    data_without_stopwords['clean_review']= data_without_stopwords['review without stopwords'].apply(lambda cw : remove_tags(cw))
    data_without_stopwords['clean_review'] = data_without_stopwords['clean_review'].str.replace('[{}]'.format(string.punctuation), ' ', regex=True)
    return data_without_stopwords

def read_glove_vector(glove_vec):
    with open(glove_vec, 'r', encoding='UTF-8') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            w_line = line.split()
            curr_word = w_line[0]
            word_to_vec_map[curr_word] = np.array(w_line[1:], dtype=np.float64)
    return word_to_vec_map

def remove_stopwords(data):
    data['review without stopwords'] = data['review'].apply(lambda x : ' '.join([word for word in x.split() if word not in (stopwords)]))
    return data

def remove_tags(string):
    result = re.sub('<.*?>','',string)
    return result

def usefulness_rating(input_shape):
    X_indices = Input(input_shape)
    embeddings = embedding_layer(X_indices)
    X = LSTM(128, return_sequences=True)(embeddings)
    X = Dropout(0.6)(X)
    X = LSTM(128, return_sequences=True)(X)
    X = Dropout(0.6)(X)
    X = LSTM(128)(X)
    X = Dense(1, activation='linear')(X)
    model = Model(inputs=X_indices, outputs=X)
    return model

if __name__ == '__main__':
    # load data
    df_train = pd.read_csv('../processed_data/train_meta_inc.csv')
    df_test = pd.read_csv('../processed_data/test_meta_inc.csv')
    df_train['review'] = df_train['review'].str.lower()
    df_test['review'] = df_test['review'].str.lower()
    
    df_train_data_without_stopwords = clean_review(df_train)
    df_test_data_without_stopwords = clean_review(df_test)

    X_train = df_train_data_without_stopwords['clean_review']
    y_train = df_train_data_without_stopwords['found_helpful_percentage']
    X_test = df_test_data_without_stopwords['clean_review']
    y_test = df_test_data_without_stopwords['found_helpful_percentage']
    

    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)
    words_to_index = tokenizer.word_index


    word_to_vec_map = read_glove_vector('../model/glove/glove.6B.50d.txt')
    maxLen = 150


    vocab_len = len(words_to_index)
    embed_vector_len = word_to_vec_map['moon'].shape[0]
    emb_matrix = np.zeros((vocab_len, embed_vector_len))
    for word, index in words_to_index.items():
        embedding_vector = word_to_vec_map.get(word)
        if embedding_vector is not None:
            emb_matrix[index, :] = embedding_vector
    embedding_layer = Embedding(input_dim=vocab_len, output_dim=embed_vector_len, input_length=maxLen, weights=[emb_matrix], trainable=False)

    X_train_indices = tokenizer.texts_to_sequences(X_train)
    X_train_indices = pad_sequences(X_train_indices, maxlen=maxLen, padding='post')
    X_test_indices = tokenizer.texts_to_sequences(X_test)
    X_test_indices = pad_sequences(X_test_indices, maxlen=maxLen, padding='post')

    adam = keras.optimizers.Adam(learning_rate = 0.0001)
    model = usefulness_rating(X_train_indices.shape[1])
    model.compile(optimizer=adam, loss='mean_absolute_error', metrics=['mean_squared_error'])
    model.fit(X_train_indices, y_train, batch_size=64, epochs=15)
    model.evaluate(X_test_indices, y_test)
    preds = model.predict(X_test_indices)

    # write results to file
    df_test = pd.read_csv('../processed_data/test_meta_inc.csv')
    if os.path.exists('../processed_data/test_pred.csv'):
        df_test = pd.read_csv('../processed_data/test_pred.csv')
    df_test_pred = df_test.copy()
    df_test_pred['pred_lstm_untuned'] = preds
    # save to disk
    out_dir = '../processed_data'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    df_test_pred.to_csv(os.path.join(out_dir, 'test_pred.csv'), index=False)