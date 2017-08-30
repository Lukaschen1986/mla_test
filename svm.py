# -*- coding: utf-8 -*-
from __future__ import division
import os
os.getcwd()
import numpy as np
#np.set_printoptions(precision=4)
import pandas as pd
import random as rd
#from sklearn import preprocessing
#from sklearn import datasets

df = pd.read_csv("swiss.csv")

X = np.array(df.iloc[:,0:5])
y = np.array(df.iloc[:,5])
y = y[:,np.newaxis]
n, p = X.shape[0], X.shape[1]

a = np.zeros((n,1))
b = 0
C = 0.6
max_iter = 40

a_change = 0
for i in range(n):
    # i = 0
    y_hat = a[i]*y[i]*X[i].dot(X[i]) + b
    Ei = y_hat - y[i]
    if y[i]*y_hat > 1 and a > 0 or y[i]*y_hat == 1 and a == 0 or a == C
