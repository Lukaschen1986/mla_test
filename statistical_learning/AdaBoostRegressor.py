# -*- coding: utf-8 -*-
# https://www.jianshu.com/p/15c82ec2d66d
from __future__ import absolute_import, division, print_function
import os
os.getcwd()
import pickle
import copy
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt

dataSet = pd.read_csv("swiss.csv", sep=",", index_col="name")
dataSet = shuffle(dataSet)
df_train, df_test = train_test_split(dataSet, test_size=0.2)

x_train = np.array(df_train.iloc[:,1:])
y_train = np.array(df_train.iloc[:,0])
x_test = np.array(df_test.iloc[:,1:])
y_test = np.array(df_test.iloc[:,0])


def init_weights(x):
    nrow = len(x)
    init_value = 1.0 / nrow
    weights = np.tile(init_value, nrow)
    return weights

def update_epsilon(weights, y_train, y_hat):
    eps_max = np.max(np.abs(y_train-y_hat))
    eps_i = np.abs(y_train - y_hat) / eps_max
    eps = np.sum(eps_i * weights)
    return eps, eps_i
  
def update_alpha(eps):
    alpha = eps/(1.0-eps)
    return alpha
  
def update_weights(weights, y_train, y_hat, alpha, eps_i):
    z = np.sum(weights * alpha**(1-eps_i))
    weights = weights * alpha**(1-eps_i) / z
    return weights
  
def fit(x_train_update, y_train, tree_depth, max_iter):
    # init_weights
    weights = init_weights(x_train)
    x_train_update = np.expand_dims(weights, axis=1) * x_train
    # bulid model
    reg_dt = DecisionTreeRegressor(criterion="mse",
                                   max_depth=tree_depth,
                                   min_samples_split=2,
                                   min_samples_leaf=1,
                                   max_features=None,
                                   random_state=0)
    reg_dt_model_res = []; eps_res = []; alpha_res = []
    
    for i in range(max_iter):
        # fit model
        reg_dt_model = reg_dt.fit(x_train_update, y_train)
        # save model
        reg_dt_model_res.append(reg_dt_model)
        # predict on train
        y_hat_train = reg_dt_model.predict(x_train_update)
        # update_epsilon
        eps, eps_i = update_epsilon(weights, y_train, y_hat_train)
        eps_res.append(eps)
        # update_alpha
        alpha = update_alpha(eps)
        alpha_res.append(alpha)
        # update_weights
        weights = update_weights(weights, y_train, y_hat_train, alpha, eps_i)
        # x_train_update
        x_train_update = np.expand_dims(weights, axis=1) * x_train
    return reg_dt_model_res, eps_res, alpha_res
  
  
def predict(x_test, reg_dt_model_res, alpha_res):
    y_pred = 0
    
    for i in range(max_iter):
        reg_dt_model = reg_dt_model_res[i]
        y_hat = reg_dt_model.predict(x_test_update)
        y_pred += np.log(1.0/alpha_res[i]) * y_hat
