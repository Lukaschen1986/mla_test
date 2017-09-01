# -*- coding: utf-8 -*-
from __future__ import division
import os
os.getcwd()
import numpy as np
#np.set_printoptions(precision=4)
import pandas as pd
import random as rd
import scipy.spatial.distance as dist
import copy
#from sklearn import preprocessing
#from sklearn import datasets

df = pd.read_csv("swiss.csv")

X = np.array(df.iloc[:,0:5])
y = np.array(df.iloc[:,5])
y = y[:,np.newaxis]
n, p = X.shape[0], X.shape[1]

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


poly_kernel = lambda X_train, X_test, sita, gamma, d: (sita + gamma*X_train.dot(X_test.T))**d
rbf_kernel = lambda X_train, X_test, gamma: np.exp(-gamma * dist.cdist(X_train, X_train)**2)

K_ploy = poly_kernel(X, X, 1, 0.001, 2)
K_rbf = rbf_kernel(X, X, 0.0001)
#np.atleast_2d(x)
#.flatten()

def find_bounds(y, i, j):
    if y[i] == y[j]:
        L = max(0, a[i] + a[j] - C)
        H = min(a[i] + a[j], C)
    else:
        L = max(0, a[j] - a[i])
        H = min(C, C - a[i] + a[j])
    return L, H
        
def clip(a, L, H):
    if a > H:
        a = H
    elif a < L:
        a = L
    else:
        a = a
    return a




a = np.zeros((n,1))
b = 0
C = 0.6
max_iters = 200
tol = 0.001

    iters = 0
    while iters < max_iters:
        iters += 1
        a_prev = copy.deepcopy(a)
        
        for i in range(n):
            Ki = poly_kernel(X, X[i], 0, 1, 1)
            ui = (a*y).T.dot(Ki) + b
            Ei = ui - y[i]
            # weifan KKT: i
            if y[i]*Ei >= 1 and a[i] > 0 or y[i]*Ei <= 1 and a[i] < C:
                # Pick random i
                i = random_idx(n, j)
                # Error for i
                Ki = poly_kernel(X, X[i], 0, 1, 1)
                ui = (a*y).T.dot(Ki) + b # y_hat
                Ei = ui - y[i]
                # 更新上下限
                L, H = find_bounds(y, i, j)
                # 计算eta
                eta = K[i,i] + K[j,j] - 2*K[i,j]
#                if eta <= 0:
#                    continue
                # Save old alphas
                ai_old, aj_old = a[i], a[j]
                # Update alpha
                a[j] = aj_old + y[j]*(Ei-Ej)/eta
                a[j] = clip(a[j], L, H)
                a[i] = ai_old + y[j]/y[i]*(aj_old-a[j])
                # Find intercept
                b1 = b - y[i] + ui - y[i]*(a[i]-ai_old)*K[i,i] - y[j]*(a[j]-aj_old)*K[i,j]
                b2 = b - y[j] + uj - y[i]*(a[i]-ai_old)*K[i,j] - y[j]*(a[j]-aj_old)*K[j,j]
                if 0 < a[i] < C:
                    b = b1
                elif 0 < a[j] < C:
                    b = b2
                else:
                    b = 0.5 * (b1 + b2)
            else:
                continue
            # Check convergence
            diff = np.sqrt(np.sum((a-a_prev)**2))
            if diff < tol:
                break
