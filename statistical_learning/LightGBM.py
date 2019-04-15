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
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
import lightgbm as lgb
from lightgbm.sklearn import LGBMRegressor
'''
# https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html
LightGBM can use categorical features as input directly. It doesn’t need to convert to one-hot coding, and is much faster than one-hot coding (about 8x speed-up).
Note: You should convert your categorical features to int type before you construct Dataset.
'''

def get_lgb_data(x_train_sparse, y_train, x_valid_sparse, y_valid, feat_names):
    d_train = lgb.Dataset(data=x_train_sparse, label=y_train, feature_name=feat_names)
    d_valid = lgb.Dataset(data=x_valid_sparse, label=y_valid, feature_name=feat_names, reference=d_train)
    valid_sets = [d_valid, d_train]
    return d_train, valid_sets


def get_best_estimators(d_train, params_lgb_1, params_lgb_2, k, verbose=True):
    # train best n_estimators
    model_cv = lgb.cv(params=params_lgb_2, 
                      train_set=d_train, 
                      num_boost_round=params_lgb_2["n_estimators"],
                      nfold=k, #k=3
                      stratified=False,
                      metrics="rmse", 
                      early_stopping_rounds=int(params_lgb_2["n_estimators"]/10),
                      verbose_eval=verbose)
    # update params
    n_estimators = len(model_cv["rmse-mean"])
    print("update n_estimators to: {}".format(n_estimators))
    params_lgb_1.set_params(n_estimators=n_estimators)
    params_lgb_2 = params_lgb_1.get_params()
    return params_lgb_1, params_lgb_2


def get_best_params(x_train_sparse, y_train, params_lgb_1, params_lgb_2, k, params_hyp):
    # train best params
    params_cv = GridSearchCV(estimator=params_lgb_1, 
                             param_grid=params_hyp, 
                             scoring="neg_mean_squared_error", 
                             cv=k, 
                             n_jobs=-1)
    model_cv = params_cv.fit(x_train_sparse, y_train)
    # update params
    params_lgb_1 = model_cv.best_estimator_
    params_lgb_2 = params_lgb_1.get_params()
    return params_lgb_1, params_lgb_2


def train_model(d_train, valid_sets, params_lgb_2, verbose):
    model_lgb = lgb.train(params=params_lgb_2,
                          train_set=d_train,
                          valid_sets=valid_sets,
                          valid_names=["valid","train"],
                          num_boost_round=params_lgb_2["n_estimators"],
                          early_stopping_rounds=int(params_lgb_2["n_estimators"]/10),
                          verbose_eval=verbose)
    return model_lgb


@caltime
def train_lgb(x_train_sparse, y_train, x_valid_sparse, y_valid, feat_names,
              n_estimators_0, objective, eval_metric, k, verbose):
    # get_lgb_data
    d_train, valid_sets = get_lgb_data(x_train_sparse, y_train, x_valid_sparse, y_valid, feat_names)
    # define init params
    params_lgb_1 = LGBMRegressor(boosting_type="gbdt",
                                 #silent=True,
                                 n_jobs=-1,
                                 learning_rate=0.1,
                                 n_estimators=n_estimators_0, #n_estimators_0=100
                                 min_gain_to_split=0, #gamma=0,
                                 #max_depth=9,
                                 num_leaves=31,
                                 min_child_weight=0.001,
                                 bagging_fraction=0.9, #subsample=0.9,
                                 feature_fraction=0.9, #colsample_bytree=0.9,
                                 lambda_l1=0, #reg_alpha=0,
                                 lambda_l2=0, #reg_lambda=1,
                                 max_delta_step=0,
                                 scale_pos_weight=1,
                                 objective=objective, #objective="regression"; objective="binary"
                                 metric=eval_metric, #eval_metric="rmse"; eval_metric="auc"
                                 seed=0)
    params_lgb_2 = params_lgb_1.get_params()
    # get_best_estimators
    params_lgb_1, params_lgb_2 = get_best_estimators(d_train, params_lgb_1, params_lgb_2, k, verbose)
    # learning_rate
    params_lgb_1, params_lgb_2 = get_best_params(x_train_sparse, y_train, params_lgb_1, params_lgb_2, k,
                                                 params_hyp=dict(learning_rate=[0.1, 0.2, 0.3]))
    print("learning_rate: ", params_lgb_2["learning_rate"])
    # num_leaves
    params_lgb_1, params_lgb_2 = get_best_params(x_train_sparse, y_train, params_lgb_1, params_lgb_2, k,
                                                 params_hyp=dict(num_leaves=[15,31,65,127]))
    print("num_leaves: {}".format(params_lgb_2["num_leaves"]))
    # min_child_weight
    params_lgb_1, params_lgb_2 = get_best_params(x_train_sparse, y_train, params_lgb_1, params_lgb_2, k,
                                                 params_hyp=dict(min_child_weight=[0.001, 0.01, 0.1, 1, 10]))
    print("min_child_weight: {}".format(params_lgb_2["min_child_weight"]))
    # min_gain_to_split / gamma
    params_lgb_1, params_lgb_2 = get_best_params(x_train_sparse, y_train, params_lgb_1, params_lgb_2, k,
                                                 params_hyp=dict(min_gain_to_split=[0, 0.5, 1, 1.5, 2, 2.5]))
    print("min_gain_to_split: ", params_lgb_2["min_gain_to_split"])
    # bagging_fraction / subsample & feature_fraction / colsample_bytree
    params_lgb_1, params_lgb_2 = get_best_params(x_train_sparse, y_train, params_lgb_1, params_lgb_2, k,
                                                 params_hyp=dict(bagging_fraction=np.arange(0.6,1.0,0.05), 
                                                                 feature_fraction=np.arange(0.6,1.0,0.05)))
    print("bagging_fraction: {0}; feature_fraction: {1}".format(params_lgb_2["bagging_fraction"],params_lgb_2["feature_fraction"]))
    # lambda_l1 / reg_alpha
    params_lgb_1, params_lgb_2 = get_best_params(x_train_sparse, y_train, params_lgb_1, params_lgb_2, k,
                                                 params_hyp=dict(lambda_l1=[0,1,2,3,5,7]))
    print("lambda_l1: ", params_lgb_2["lambda_l1"])
    # lambda_l2 / reg_lambda
    params_lgb_1, params_lgb_2 = get_best_params(x_train_sparse, y_train, params_lgb_1, params_lgb_2, k,
                                                 params_hyp=dict(lambda_l2=[0,1,3,5,7,9]))
    print("lambda_l2: ", params_lgb_2["lambda_l2"])
    # max_delta_step & scale_pos_weight
    params_lgb_1, params_lgb_2 = get_best_params(x_train_sparse, y_train, params_lgb_1, params_lgb_2, k,
                                                 params_hyp=dict(max_delta_step=[0, 1, 3], 
                                                                 scale_pos_weight=[1, 3, 5]))
    print("max_delta_step: {0}; scale_pos_weight: {1}".format(params_lgb_2["max_delta_step"],params_lgb_2["scale_pos_weight"]))
    # get_best_estimators
    params_lgb_1, params_lgb_2 = get_best_estimators(d_train, params_lgb_1, params_lgb_2, k, verbose)
    # train_model
    model_lgb = train_model(d_train, valid_sets, params_lgb_2, verbose)
    print("LGB model train complete.")
    # save model
    model_lgb.save_model(filename="model_lgb.model")
    print("model has saved.")
    # load model
    #model_lgb = lgb.Booster(model_file="model_lgb.model")
    # with open(workpath + "/data/models/{0}/model_lgb_{1}.txt".format(self.datasource,self.end_time), "wb") as f:
    # pickle.dump(model_lgb, f)
    # with open(workpath + "/data/hyper/{}/params_lgb_1.txt".format(self.datasource), "wb") as f:
    # pickle.dump(params_lgb_1, f)
    # print("param has saved.")
    return model_lgb

if __name__ == "__main__":
    dataSet = pd.read_csv("swiss2.csv")
    dataSet = shuffle(dataSet, random_state=0)
    df_train, df_test = train_test_split(dataSet, test_size=0.2, random_state=0)
    
    x_train = df_train.iloc[:,2:]
    y_train = df_train.iloc[:,1]
#    y_train = np.log1p(y_train.values)
    
    x_test = df_test.iloc[:,2:]
    y_test = df_test.iloc[:,1]
#    y_test = np.log1p(y_test.values)
    
    feat_names = x_train.columns

    # 转为稀疏矩阵
    x_train_sparse = sparse.csr_matrix(x_train)
    x_test_sparse = sparse.csr_matrix(x_test)    
    ......
    
    model_dump = model_lgb.dump_model(num_iteration=-1)

    #df_feats = pd.DataFrame({"feat_names": feat_names,
    #                         "importances": model_lgb.feature_importance(importance_type="split")})
    #df_feats = df_feats.set_index("feat_names").sort_values(by="importances", ascending=True)
    #df_feats.plot(kind="barh", title="feature_importances", grid=True, figsize=(12,31))
    lgb.plot_importance(model_lgb, max_num_features=30, importance_type="split",
                        figsize=(12,31), title="LightGBM Feature Importance")
    lgb.plot_tree(model_lgb, tree_index=0, figsize=(12,31))
        
    
