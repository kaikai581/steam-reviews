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
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression

import argparse
import numpy as np
import os
import pandas as pd
import pickle as pkl
import smogn
import sys
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
    parser.add_argument('-d', '--glove_dim', type=int, default=50)
    parser.add_argument('-m', '--model', type=int, default=0,
        help=' \
            0: untuned RandomForestRegressor\n \
            1: untuned XGBoost \
        '
    )
    parser.add_argument('--imbalanced', action='store_true')
    parser.add_argument('--load_weight', action='store_true')
    args = parser.parse_args()
    regr_opts = {
        0: ('pred_rfr_untuned', RandomForestRegressor),
        1: ('pred_xgbr_untuned', xgboost.XGBRegressor),
        2: ('pred_krr_untuned', KernelRidge),
        3: ('pred_linr_untuned', LinearRegression)
    }

    if not args.glove_dim in [50, 100, 200, 300]:
        print('GloVe dimension must be one in {50, 100, 200, 300}.')
        sys.exit(-1)

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
    max_length = args.glove_dim
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
    if args.imbalanced:
        df_temp = pd.DataFrame(X_train_vec)
        df_temp['label'] = y_train
        df_smogn = smogn.smoter(
            data = df_temp, 
            y = 'label'
        )
        y_train = df_smogn.label
        X_train_vec = df_smogn.drop(columns='label')

    # train an untuned random forest regressor
    pred_colname, regr = regr_opts[args.model]
    if args.model == 0:
        # regressor = regr(bootstrap=True, max_depth=10, max_features='sqrt', min_samples_leaf=4,
        #                  min_samples_split=2, n_estimators=400)
        regressor = regr()
    elif args.model == 1:
        '''
        dim=50
        Best parameters found with the xgb_hp_tuning.py script:
        Fitting 5 folds for each of 288 candidates, totalling 1440 fits
        {'colsample_bytree': 0.5,
        'learning_rate': 0.01,
        'max_depth': 3,
        'min_child_weight': 1,
        'n_estimators': 500,
        'objective': 'reg:squarederror',
        'subsample': 0.5}
        '''
        '''
        dim=300
        Best parameters found with the xgb_hp_tuning.py script:
        Fitting 5 folds for each of 288 candidates, totalling 1440 fits
        {'colsample_bytree': 0.7,
        'learning_rate': 0.01,
        'max_depth': 3,
        'min_child_weight': 1,
        'n_estimators': 500,
        'objective': 'reg:squarederror',
        'subsample': 0.5}
        '''
        regressor = regr(n_estimators=500, max_depth=3, colsample_bytree=0.7, learning_rate=0.01, min_child_weight=1, subsample=0.7)
    elif args.model == 2:
        regressor = regr(alpha=1)
    elif args.model == 3:
        regressor = regr()
    has_weight_file = os.path.exists('../processed_data/train_sample_weight.csv') and args.load_weight
    if has_weight_file:
        df_weight = pd.read_csv('../processed_data/train_sample_weight.csv')
    print('===== start fitting the model =====')
    if not has_weight_file:
        regressor.fit(X_train_vec, y_train)
    else:
        regressor.fit(X_train_vec, y_train, sample_weight=df_weight.sample_weight)
    print('===== end fitting the model =====')
    print('===== start predicting with the model =====')
    y_test_pred = regressor.predict(X_test_vec)
    print('===== end predicting with the model =====')
    # save the preprocessed training data to file
    train_fpn = '../processed_data/train_glove{}d.pkl'.format(args.glove_dim)
    if args.imbalanced:
        train_fpn = '../processed_data/train_smogn.pkl'
    with open(train_fpn, 'wb') as f:
        pkl.dump([X_train_vec, y_train], f)
    '''
    To load the pickled data, do the following.
    ref: https://stackoverflow.com/questions/44466993/python-how-to-save-training-datasets
    with open(train_fpn, 'rb') as f:
        X_train_vec, y_train = pkl.load(f)
    '''
    test_fpn = '../processed_data/test_glove{}d.pkl'.format(args.glove_dim)
    with open(test_fpn, 'wb') as f:
        pkl.dump([X_test_vec, y_test], f)

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
    df_test_pred.to_csv(os.path.join(out_dir, 'test_pred_glove{}d.csv'.format(args.glove_dim)), index=False)
