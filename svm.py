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
max_iters = 40

a_change = 0
for i in range(n):
    # i = 0
    y_hat = a[i]*y[i]*X[i].dot(X[i]) + b
    Ei = y_hat - y[i]
    # 违反KKT
    if y[i]*y_hat > 1 and a[i] > 0 or y[i]*y_hat < 1 and a[i] < C:
        
        

def random_idx(n, j):
    i = j
    while i == j:
        i = np.random.randint(0,n)
    return i
        
K = (1 + 0.001*X.dot(X.T))**2

def find_bounds(y, i, j):
    if y[i] == y[j]:
        L = max(0, a[i] + a[j] - C)
        H = min(a[i] + a[j], C)
    else:
        L = max(0, a[j] - a[i])
        H = min(C, C - a[i] + a[j])
    return L, H
        
np.arange(0,n)

def train():
    iters = 0
    while iters < max_iters:
        iters += 1
        a_prev = np.copy(a)
    
    for j in range(n): # j = 0
        # Pick random i
        i = random_idx(n, j)
        
        eta = K[i,i] + K[j,j] - 2*K[i,j]
        if eta <= 0:
            continue
        L, H = find_bounds(y, i, j)
        
        # Error for current examples
        y_hat = a[i]*y[i]*X[i].dot(X.T) + b

np.random.randint(0,n-1)
