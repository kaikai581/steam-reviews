#!/usr/bin/env python
'''
This script calculates a mean GloVe vector for each review
and trains a regression model to predict the usefulness
of each review.

This script is directly adapted from the jupyter notebook "revisit_nlp.ipynb".
'''

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.ensemble import RandomForestRegressor

import argparse
import numpy as np
import os
import pandas as pd
import xgboost


# define a transformer class to engineer features
from sklearn.base import BaseEstimator, TransformerMixin

class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        print('Loading in word vectors...')
        self.word_vectors = model
        print('Finished loading in word vectors')
    def fit(self, data):
        return self
    def transform(self, data):
        # determine the dimensionality of vectors
        v = self.word_vectors.get_vector('king')
        self.D = v.shape[0]

        X = np.zeros((len(data), self.D))
        n = 0
        emptycount = 0
        for sentence in data:
            tokens = sentence.split()
            vecs = []
            m = 0
            for word in tokens:
                try:
                    # throws KeyError if word not found
                    vec = self.word_vectors.get_vector(word)
                    vecs.append(vec)
                    m += 1
                except KeyError:
                    pass
            if len(vecs) > 0:
                vecs = np.array(vecs)
                X[n] = vecs.mean(axis=0)
            else:
                emptycount += 1
            n += 1
        print('Numer of samples with no words found: %s / %s' % (emptycount, len(data)))
        return X


if __name__ == '__main__':
    # command line options
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=int, default=0,
        help=' \
            0: untuned RandomForestRegressor\n \
            1: untuned XGBoost \
        '
    )
    args = parser.parse_args()
    regr_opts = {
        0: ('pred_rfr_untuned', RandomForestRegressor),
        1: ('pred_xgbr_untuned', xgboost.XGBRegressor)
    }

    # load train and test data
    df_train = pd.read_csv('../processed_data/train_meta_inc.csv')
    df_test = pd.read_csv('../processed_data/test_meta_inc.csv')
    if os.path.exists('../processed_data/test_pred.csv'):
        df_test = pd.read_csv('../processed_data/test_pred.csv')

    # single out written reviews and labels
    X_train, y_train = df_train['review'], df_train['found_helpful_percentage']
    X_test, y_test = df_test['review'], df_test['found_helpful_percentage']

    ##########################
    ### data preprocessing ###
    ##########################
    # (hyper)parameter settings
    # ref: https://heartbeat.comet.ml/text-classification-using-long-short-term-memory-glove-embeddings-6894abb730e1
    vocab_size = 1000
    oov_token = '<OOV>'
    max_length = 100
    padding_type = 'post'
    truncation_type = 'post'

    tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(X_train)

    word_index = tokenizer.word_index
    X_train_sequences = tokenizer.texts_to_sequences(X_train)
    X_train_padded = pad_sequences(X_train_sequences, maxlen=max_length, padding=padding_type, truncating=truncation_type)

    X_test_sequences = tokenizer.texts_to_sequences(X_test)
    X_test_padded = pad_sequences(X_test_sequences, maxlen=max_length, padding=padding_type, truncating=truncation_type)

    # download the pretrained GloVe pre-trained word vectors if not exists
    # $ conda install -c conda-forge python-wget
    import wget
    model_url = 'https://nlp.stanford.edu/data/glove.6B.zip'
    out_dir = '../model/glove'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_fpn = os.path.join(out_dir, 'glove.6B.zip')
    if not os.path.exists(out_fpn):
        model_file = wget.download(model_url, out=out_fpn)
    
    # obtain the token for the zip file without unzip
    import zipfile
    unzipped_model = zipfile.ZipFile(out_fpn, 'r')
    print(unzipped_model.namelist())

    # extract the single file for use
    # since extracting on the fly takes much much time...
    ext_file = f'glove.6B.{max_length}d.txt'
    ext_model_fpn = os.path.join(out_dir, ext_file)
    if not os.path.exists(ext_model_fpn):
        unzipped_model.extract(ext_file, out_dir)
    
    # load the pretrained vector
    embeddings_index = {}
    f = open(ext_model_fpn)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))

    # construct the embedding matrix
    embedding_matrix = np.zeros((len(word_index) + 1, max_length))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    
    '''
    Use third party utilities.
    Ref:
    https://edumunozsala.github.io/BlogEms/jupyter/nlp/classification/embeddings/python/2020/08/15/Intro_NLP_WordEmbeddings_Classification.html
    '''
    from gensim.scripts.glove2word2vec import glove2word2vec
    from pathlib import Path
    word2vec_output_fpn = os.path.join('../model/glove', Path(ext_model_fpn).stem + '.w2v')
    if not os.path.exists(word2vec_output_fpn):
        glove2word2vec(ext_model_fpn, word2vec_output_fpn)
    
    # play around with the GloVe embedding
    # load the converted word2vec model from glove
    from gensim.models import KeyedVectors
    model = KeyedVectors.load_word2vec_format(word2vec_output_fpn, binary=False)
    # play around a bit
    # Show a word embedding
    print('King: ',model.get_vector('king'))
    result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
    print('Most similar word to King + Woman: ', result)

    # print one sample review
    print(len(X_train),len(X_test))
    for review in X_train:
        print(review)
        break

    # Set a word vectorizer
    vectorizer = Word2VecVectorizer(model)
    # Get the review embeddings for the train dataset
    X_train_vec = vectorizer.fit_transform(list(X_train))
    X_test_vec = vectorizer.fit_transform(list(X_test))

    # train an untuned random forest regressor
    pred_colname, regr = regr_opts[args.model]
    if args.model == 0:
        regressor = regr()
    else:
        regressor = regr(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
    print('===== start fitting the model =====')
    regressor.fit(X_train_vec, y_train)
    print('===== end fitting the model =====')
    print('===== start predicting with the model =====')
    y_test_pred = regressor.predict(X_test_vec)
    print('===== end predicting with the model =====')

    # store the predicted results
    from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
    y_train_pred = regressor.predict(X_train_vec)
    print('$r^2$', r2_score(y_test, y_test_pred))
    print('mean absolute error', mean_absolute_error(y_test, y_test_pred))
    print('mean absolute percentage error', mean_absolute_percentage_error(y_test, y_test_pred))

    # store the predicted score
    df_test_pred = df_test.copy()
    df_test_pred[pred_colname] = y_test_pred

    # save to disk
    out_dir = '../processed_data'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    df_test_pred.to_csv(os.path.join(out_dir, 'test_pred.csv'), index=False)
