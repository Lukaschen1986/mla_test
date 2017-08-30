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
        
def clip(a, L, H):
    if a > H:
        a = H
    elif a < L:
        a = L
    else:
        a = a
    return a

tol = 0.001

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
        Ki = (1 + 0.001*X.dot(X[i].T))**2
        y_hat_i = (a*y).T.dot(Ki) + b
        Kj = (1 + 0.001*X.dot(X[j].T))**2        
        y_hat_j = (a*y).T.dot(Kj) + b
        Ei = y_hat_i - y[i]
        Ej = y_hat_j - y[j]    
        
        # Save old alphas
        ai_old, aj_old = a[i], a[j]
        
        # Update alpha
        a[j] = aj_old + y[j]*(Ei-Ej)/eta
        a[j] = clip(a[j], L, H)
        a[i] = ai_old + y[j]/y[i]*(aj_old-a[j])
        
        # Find intercept
        b1 = b - y[i] + y_hat_i - y[i]*(a[i]-ai_old)*K[i,i] - y[j]*(a[j]-aj_old)*K[i,j]
        b2 = b - y[j] + y_hat_j - y[i]*(a[i]-ai_old)*K[i,j] - y[j]*(a[j]-aj_old)*K[j,j]
        if 0 < a[i] < C:
            b = b1
        elif 0 < a[j] < C:
            b = b2
        else:
            b = 0.5 * (b1 + b2)
        
        # Check convergence
        diff = np.sqrt(np.sum((a-a_prev)**2))
        if diff < tol:
            break
        
    # Save support vectors index
    sv_idx = np.where(a > 0)[0]

        
