#!/usr/bin/env python

from pprint import pprint
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

import argparse
import os
import pickle as pkl
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--glove_dim', type=int, default=50)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    # load training data
    train_fpn = '../processed_data/train_glove{}d.pkl'.format(args.glove_dim)
    if not os.path.exists(train_fpn):
        print(f'Input file {train_fpn} does not exist.')
        sys.exit(-1)
    with open(train_fpn, 'rb') as f:
        X_train_vec, y_train = pkl.load(f)

    print('Input shape:', X_train_vec.shape)

    param_tuning = {
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5, 7, 10],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.5, 0.7],
        'colsample_bytree': [0.5, 0.7],
        'n_estimators' : [100, 200, 500],
        'objective': ['reg:squarederror']
    }

    if not args.gpu:
        xgb_model = XGBRegressor()
    else:
        xgb_model = XGBRegressor(tree_method='gpu_hist')

    gsearch = GridSearchCV(estimator = xgb_model,
                           param_grid = param_tuning,                        
                           #scoring = 'neg_mean_absolute_error', #MAE
                           #scoring = 'neg_mean_squared_error',  #MSE
                           cv = 5,
                           n_jobs = None if args.gpu else -1, # use default if GPU is used
                           verbose = 1)

    gsearch.fit(X_train_vec, y_train)

    pprint(gsearch.best_params_)