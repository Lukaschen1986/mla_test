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


weights = init_weights(x_train)
x_train_update = np.expand_dims(weights, axis=1) * x_train


dtr = DecisionTreeRegressor(criterion="mse",
                            max_depth=4,
                            min_samples_split=2,
                            min_samples_leaf=1,
                            max_features="sqrt",
                            random_state=0)
reg_dtr_model = dtr.fit(x_train_update, y_train)
y_hat_train = reg_dtr_model.predict(x_train_update)

eps_max = np.max(np.abs(y_train-y_hat_train))

#eps_i = 1 - np.exp((-y_train+y_hat_train) / eps_max)
#eps_i = (y_train - y_hat_train)**2 / eps_max
eps_i = np.abs(y_train - y_hat_train) / eps_max


eps = eps_i.dot(weights)

alpha = eps / (1-eps)

z = alpha**(1-eps_i).dot(weights)

weights = weights * alpha**(1-eps_i) / z

