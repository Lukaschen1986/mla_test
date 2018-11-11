# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import os
os.getcwd()
import pickle
import copy
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
import tushare as ts
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

txt_path = "D:\\my_project\\Python_Project\\quant\\"

types = {"mpg": np.int64,
         "disp": np.float64,
         "hp": np.float64,
         "drat": np.float64,
         "wt": np.float64,
         "qsec": np.int64,
         "vs": np.int64,
         "am": np.int64,
         "gear": np.int64,
         "carb": np.int64,
         "cyl": np.int64}

df = pd.read_csv(txt_path + "mtcars2.csv")
df.vs = df.vs.astype(np.uint8)
df.am = df.am.astype(np.uint8)
df.hp = df.hp.astype(np.float64)
df = shuffle(df, random_state=0)

y = df.iloc[:,0].values
x = df.iloc[:,1:]
x = pd.get_dummies(x, columns=["gear","carb","cyl"], drop_first=False).values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

#avg = np.mean(x_train[["disp","hp","drat","wt","qsec"]], axis=0)
#std = np.std(x_train[["disp","hp","drat","wt","qsec"]], axis=0)
#x_train[["disp","hp","drat","wt","qsec"]] = (x_train[["disp","hp","drat","wt","qsec"]] - avg) / std
#x_test[["disp","hp","drat","wt","qsec"]] = (x_test[["disp","hp","drat","wt","qsec"]] - avg) / std

avg = np.mean(x_train[:,0:5], axis=0)
std = np.std(x_train[:,0:5], axis=0)
x_train[:,0:5] = (x_train[:,0:5] - avg) / std
x_test[:,0:5] = (x_test[:,0:5] - avg) / std

nrow, ncol = x_train.shape
k = 2
v = np.random.rand(ncol, k)
#v = np.random.randn(ncol, k)
#arrayvt = np.random.rand(k, len(x_train.columns))
#v = pd.DataFrame(arrayv, columns=range(0,k), index=x_train.columns)
#vt = pd.DataFrame(arrayvt, columns=x_train.columns, index=range(0,k))

w = np.zeros((ncol, 1)) + 0.001
b = np.zeros((1, 1))

lost_log = []
y = np.expand_dims(y_train, axis=1)
#y = y_train
x = x_train

unit_vector = lambda x: x / np.sqrt(np.sum(x**2, axis=0))
eta = 0.01
tol = 10**-8
max_iter = 5000
lam = 0.001

#b + x.dot(w) + np.sum(x.dot(v)**2-(x**2).dot(v**2), axis=1) / 2.0

#z = b + x.dot(w).T + np.sum(x.dot(v)**2-(x**2).dot(v**2), axis=1) / 2.0
#
#np.log(1 + np.exp(-y * x.dot(w)))
#np.sum(np.log(1 + np.exp(-y*z))) / nrow + lam*w.T.dot(w)[0][0]/nrow/2.0

for i in range(max_iter):
#    i = 0
    z = b + x.dot(w).T + np.sum(x.dot(v)**2-(x**2).dot(v**2), axis=1) / 2.0
    
    
    
    gb = -y.T.dot(1/(1+np.exp(y*x.dot(w)))) / nrow
    
    gw = (-y*x).T.dot(1/(1+np.exp(y*x.dot(w)))) / nrow + lam*w/nrow
    gw = unit_vector(gw)
    
    gv = x.T.dot(x.dot(v) - (x**2).dot(v))
    gv = unit_vector(gv)
    
    b -= eta * gb
    w -= eta * gw
    v -= eta * gv    
    
    z = b + x.dot(w).T + np.sum(x.dot(v)**2-(x**2).dot(v**2), axis=1) / 2.0
    lost = np.sum(np.log(1 + np.exp(-y*z))) / nrow + lam*w.T.dot(w)[0][0]/nrow/2.0
    lost_log.append(lost)
    # break condition
    if (i >= 1) and (np.abs(lost_log[i]-lost_log[i-1]) <= tol):
        break


plt.plot(lost_log)
min(lost_log)

std_scale = StandardScaler()
std_scale_fit = std_scale.fit(x_train[["disp","hp","drat","wt"]])
std_scale_fit.transform(x_train[["disp","hp","drat","wt"]])




