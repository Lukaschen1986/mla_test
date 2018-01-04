# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import os
os.getcwd()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from scipy.stats import itemfreq
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
import copy

diabetes = datasets.load_diabetes()
x = diabetes["data"]
y = diabetes["target"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# RF+GBDT
resid = copy.deepcopy(y_train)
epochs = 50; eta = 0.01
models = []; mse_res = []; 
y_actual = np.zeros_like(resid)[:,np.newaxis]
y_hat = np.zeros_like(resid)[:,np.newaxis]
bias_res = []; var_res = []

clf = RandomForestRegressor(n_estimators=100, max_features="sqrt", max_depth=None, min_samples_split=2, min_samples_leaf=1)

#epoch = 1
for epoch in range(epochs):
    clf.fit(x_train, resid)
    joblib.dump(clf, filename="./clf/clf_" + str(epoch))
    pred = clf.predict(x_train)
    
    y_hat = np.concatenate((y_hat, pred[:,np.newaxis]), axis=1) # 储存每一轮y的预测值
    y_actual = np.concatenate((y_actual, resid[:,np.newaxis]), axis=1) # 储存每一轮y的实际值
    if epoch == 0:
        y_hat = y_hat[:,1:]
        y_actual = y_actual[:,1:]
    
    resid -= eta * pred # 更新残差向量
    y_hat_final = y_hat[:,0] + np.sum(eta * y_hat[:,1:], axis=1, keepdims=False)
    
    mse = np.sum((y_train-y_hat_final)**2) / len(y_train)
    mse_res.append(mse)
    
    bias = np.mean((np.mean(y_actual, axis=1, keepdims=False) - np.mean(y_hat, axis=1, keepdims=False))**2)
    bias_res.append(bias)

    var = np.mean(np.var(y_hat, axis=1, keepdims=False))
    var_res.append(var)

metric = pd.DataFrame({"bias":bias_res, "var":var_res})
metric["mse"] = metric["bias"] + metric["var"]
metric.plot()
epochs_predict = np.argmin(metric["mse"])
print(epochs_predict)
plt.plot(mse_res)

for epoch in range(epochs_predict):
    clf = joblib.load(filename="./clf/clf_" + str(epoch))
    if epoch == 0:
        y_predict = clf.predict(x_test)
    else:
        y_predict += eta * clf.predict(x_test)
    
np.sum((y_test-y_predict)**2) / len(y_test)


# CART
tune = {"criterion": ["mse"],
        "min_samples_split": [2, 10, 20],
        "max_depth": [None, 2, 5, 10],
        "min_samples_leaf": [1, 5, 10],
        "max_leaf_nodes": [None, 5, 10, 20]}
clf = DecisionTreeRegressor()
model_fit = clf.fit(x_train, y_train)
model_fit.predict(x_train)


clf = DecisionTreeRegressor()
clf = GridSearchCV(clf, param_grid=tune, cv=10)
clf.fit(x_train, y_train)

y_hat = clf.predict(x_train)
cart_mse_train = np.sum((y_train-y_hat)**2) / len(y_train)
print("cart_mse_train: " + str(cart_mse_train))

y_pred = clf.predict(x_test)
cart_mse_test = np.sum((y_test-y_pred)**2) / len(y_test)
print("cart_mse_test: " + str(cart_mse_test))

# GBDT
clf = GradientBoostingRegressor()
#clf = GridSearchCV(clf, param_grid=tune, cv=10)
clf.fit(x_train, y_train)

y_hat = clf.predict(x_train)
gbdt_mes_train = np.sum((y_train-y_hat)**2) / len(y_train)
print("gbdt_mes_train: " + str(gbdt_mes_train))

y_pred = clf.predict(x_test)
gbdt_mse_test = np.sum((y_test-y_pred)**2) / len(y_test)
print("gbdt_mse_test: " + str(gbdt_mse_test))

# XGBoost
clf = XGBRegressor()
clf.fit(x_train, y_train)

y_hat = clf.predict(x_train)
xgboost_mes_train = np.sum((y_train-y_hat)**2) / len(y_train)

y_pred = clf.predict(x_test)
xgboost_mes_test = np.sum((y_test-y_pred)**2) / len(y_test)

# RF
clf = RandomForestRegressor(n_estimators=100, max_features="sqrt", max_depth=None, min_samples_split=2, min_samples_leaf=1)
clf.fit(x_train, y_train)

y_hat = clf.predict(x_train)
rf_mes_train = np.sum((y_train-y_hat)**2) / len(y_train) # 439.80217138810195

y_pred = clf.predict(x_test)
rf_mes_test = np.sum((y_test-y_pred)**2) / len(y_test) # 3660


