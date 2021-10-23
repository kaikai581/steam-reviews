#!/usr/bin/env python

from pprint import pprint
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

import pickle as pkl

if __name__ == '__main__':
    # load training data
    train_fpn = '../processed_data/train.pkl'
    with open(train_fpn, 'rb') as f:
        X_train_vec, y_train = pkl.load(f)

    param_tuning = {
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5, 7, 10],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.5, 0.7],
        'colsample_bytree': [0.5, 0.7],
        'n_estimators' : [100, 200, 500],
        'objective': ['reg:squarederror']
    }

    xgb_model = XGBRegressor()

    gsearch = GridSearchCV(estimator = xgb_model,
                           param_grid = param_tuning,                        
                           #scoring = 'neg_mean_absolute_error', #MAE
                           #scoring = 'neg_mean_squared_error',  #MSE
                           cv = 5,
                           n_jobs = -1,
                           verbose = 1)

    gsearch.fit(X_train_vec, y_train)

    pprint(gsearch.best_params_)