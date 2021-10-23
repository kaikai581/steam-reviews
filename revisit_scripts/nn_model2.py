#!/usr/bin/env python
'''
This script is based on the website.
https://medium.com/@sarin.samarth07/glove-word-embeddings-with-keras-python-code-52131b0c8b1d
'''

import argparse
import os
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout,Embedding,Bidirectional
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--load_weight', action='store_true')
args = parser.parse_args()

df = pd.read_csv('../processed_data/train_meta_inc.csv')
x = df['review']
y = df['found_helpful_percentage']

token = Tokenizer()
token.fit_on_texts(x)
seq = token.texts_to_sequences(x)
pad_seq = pad_sequences(seq, maxlen=300)

vocab_size = len(token.word_index)+1

embedding_vector = {}
f = open('../model/glove/glove.6B.300d.txt')
for line in tqdm(f):
    value = line.split(' ')
    word = value[0]
    coef = np.array(value[1:],dtype = 'float32')
    embedding_vector[word] = coef

embedding_matrix = np.zeros((vocab_size,300))
for word,i in tqdm(token.word_index.items()):
    embedding_value = embedding_vector.get(word)
    if embedding_value is not None:
        embedding_matrix[i] = embedding_value

has_weight_file = os.path.exists('../processed_data/train_sample_weight.csv') and args.load_weight
if has_weight_file:
    df_weight = pd.read_csv('../processed_data/train_sample_weight.csv')

model = Sequential()
model.add(Embedding(vocab_size,300,weights = [embedding_matrix],input_length=300,trainable = False))
model.add(Bidirectional(LSTM(75)))
model.add(Dense(32, activation='linear'))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
if not has_weight_file:
    history = model.fit(pad_seq,y,epochs = 5,batch_size=256,validation_split=0.2)
else:
    history = model.fit(pad_seq,y,epochs = 5,batch_size=256,validation_split=0.2, sample_weight=df_weight.sample_weight)

testing = pd.read_csv('../processed_data/test_meta_inc.csv')
x_test = testing['review']
x_test = token.texts_to_sequences(x_test)
testing_seq = pad_sequences(x_test, maxlen=300)

predict = model.predict(testing_seq)
# write results to file
df_test = pd.read_csv('../processed_data/test_meta_inc.csv')
if os.path.exists('../processed_data/test_pred.csv'):
    df_test = pd.read_csv('../processed_data/test_pred.csv')
df_test_pred = df_test.copy()
df_test_pred['pred_lstm2_untuned'] = predict
# save to disk
out_dir = '../processed_data'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
df_test_pred.to_csv(os.path.join(out_dir, 'test_pred.csv'), index=False)