# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from scipy.stats import skew, kurtosis
from scipy import sparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import skew, kurtosis
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, StandardScaler, MinMaxScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from xgboost.sklearn import XGBRegressor
import xgboost as xgb


def get_xgb_data(x_train_sparse, y_train, x_test_sparse, y_test):
    d_train = xgb.DMatrix(x_train_sparse, y_train)
    d_test = xgb.DMatrix(x_test_sparse, y_test)
    watchlist = [(d_train, "train"), (d_test, "test")]
    return d_train, watchlist


def get_best_estimators(d_train, params_xgb_1, params_xgb_2, k, verbose=True):
    # train best n_estimators
    model_cv = xgb.cv(params=params_xgb_2, 
                      dtrain=d_train, 
                      num_boost_round=params_xgb_2["n_estimators"],
                      nfold=k, 
                      metrics="rmse", 
                      early_stopping_rounds=int(params_xgb_2["n_estimators"]/10),
                      verbose_eval=verbose)
    # update params
    n_estimators = model_cv.shape[0]
    print("update n_estimators to: {}".format(n_estimators))
    params_xgb_1.set_params(n_estimators=n_estimators)
    params_xgb_2 = params_xgb_1.get_params()
    return params_xgb_1, params_xgb_2


def get_best_params(x_train_sparse, y_train, params_xgb_1, params_xgb_2, k, params_hyp):
    # train best params
    params_cv = GridSearchCV(estimator=params_xgb_1, 
                             param_grid=params_hyp, 
                             scoring="neg_mean_squared_error", 
                             cv=k, 
                             n_jobs=-1)
    model_cv = params_cv.fit(x_train_sparse, y_train)
    # update params
    params_xgb_1 = model_cv.best_estimator_
    params_xgb_2 = params_xgb_1.get_params()
    return params_xgb_1, params_xgb_2


def train_model(d_train, watchlist, params_xgb_2, verbose):
    model_xgb = xgb.train(params=params_xgb_2,
                          dtrain=d_train,
                          evals=watchlist,
                          num_boost_round=params_xgb_2["n_estimators"],
                          early_stopping_rounds=int(params_xgb_2["n_estimators"]/10),
                          verbose_eval=verbose)
    return model_xgb


def train(x_train_sparse, y_train, x_test_sparse, y_test, n_estimators_0, objective, eval_metric, k, verbose):
    # get_xgb_data
    d_train, watchlist = get_xgb_data(x_train_sparse, y_train, x_test_sparse, y_test)
    # define init params
    params_xgb_1 = XGBRegressor(booster="gbtree",
                                silent=1,
                                nthread=-1,
                                n_jobs=-1,
                                learning_rate=0.1,
                                n_estimators=n_estimators_0,
                                gamma=0,
                                max_depth=9,
                                min_child_weight=0.001,
                                subsample=0.9,
                                colsample_bytree=0.9,
                                reg_alpha=0,
                                reg_lambda=1,
                                max_delta_step=0,
                                scale_pos_weight=1,
                                objective=objective,
                                eval_metric=eval_metric,
                                seed=0)
    params_xgb_2 = params_xgb_1.get_params()
    # get_best_estimators
    params_xgb_1, params_xgb_2 = get_best_estimators(d_train, params_xgb_1, params_xgb_2, k, verbose)
    # learning_rate
    params_xgb_1, params_xgb_2 = get_best_params(x_train_sparse, y_train, 
                                                      params_xgb_1, params_xgb_2, k,
                                                      params_hyp=dict(learning_rate=[0.1, 0.2, 0.3]))
    print("learning_rate: ", params_xgb_2["learning_rate"])
    # max_depth & min_child_weight
    params_xgb_1, params_xgb_2 = get_best_params(x_train_sparse, y_train, 
                                                      params_xgb_1, params_xgb_2, k,
                                                      params_hyp=dict(max_depth=[5,7,9,11], 
                                                                 min_child_weight=[0.001, 0.01, 0.1, 1, 10]))
    print("max_depth: {0}; min_child_weight: {1}".format(params_xgb_2["max_depth"], params_xgb_2["min_child_weight"]))
    # gamma
    params_xgb_1, params_xgb_2 = get_best_params(x_train_sparse, y_train, 
                                                      params_xgb_1, params_xgb_2, k,
                                                      params_hyp=dict(gamma=[0, 0.5, 1, 1.5, 2, 2.5]))
    print("gamma: ", params_xgb_2["gamma"])
    # subsample & colsample_bytree
    params_xgb_1, params_xgb_2 = get_best_params(x_train_sparse, y_train, 
                                                      params_xgb_1, params_xgb_2, k,
                                                      params_hyp=dict(subsample=[0.6,0.7,0.8,0.9], 
                                                                      colsample_bytree=[0.6,0.7,0.8,0.9]))
    print("subsample: {0}; colsample_bytree: {1}".format(params_xgb_2["subsample"],params_xgb_2["colsample_bytree"]))
    # reg_alpha
    params_xgb_1, params_xgb_2 = get_best_params(x_train_sparse, y_train, 
                                                      params_xgb_1, params_xgb_2, k,
                                                      params_hyp=dict(reg_alpha=[0,1,2,3,5,7]))
    print("reg_alpha: ", params_xgb_2["reg_alpha"])
    # reg_lambda
    params_xgb_1, params_xgb_2 = get_best_params(x_train_sparse, y_train, 
                                                      params_xgb_1, params_xgb_2, k,
                                                      params_hyp=dict(reg_lambda=[0,1,3,5,7,9]))
    print("reg_lambda: ", params_xgb_2["reg_lambda"])
    # max_delta_step & scale_pos_weight
    params_xgb_1, params_xgb_2 = get_best_params(x_train_sparse, y_train, 
                                                      params_xgb_1, params_xgb_2, k,
                                                      params_hyp=dict(max_delta_step=[0, 1, 3], 
                                                                      scale_pos_weight=[1, 3, 5]))
    print("max_delta_step: {0}; scale_pos_weight: {1}".format(params_xgb_2["max_delta_step"],params_xgb_2["scale_pos_weight"]))
    # get_best_estimators
    params_xgb_1, params_xgb_2 = get_best_estimators(d_train, params_xgb_1, params_xgb_2, k, verbose)
    # train_model
    model_xgb = train_model(d_train, watchlist, params_xgb_2, verbose)
    print("XGB model train complete.")
    # save model
#    with open(workpath + "/data/models/{0}/model_xgb_{1}.txt".format(self.datasource,self.end_time), "wb") as f:
#        pickle.dump(model_xgb, f)
#    print("model has saved.")
#    with open(workpath + "/data/hyper/{}/params_xgb_1.txt".format(self.datasource), "wb") as f:
#        pickle.dump(params_xgb_1, f)
#    print("param has saved.")
    return model_xgb
