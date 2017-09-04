# -*- coding: utf-8 -*-
from __future__ import division
import os
os.getcwd()
import copy
import numpy as np
#np.set_printoptions(precision=4)
import pandas as pd
import random as rd
import scipy.spatial.distance as dist
from sklearn.cross_validation import train_test_split

df = pd.read_csv("swiss.csv")
X = np.array(df.iloc[:,0:5])
y = np.array(df.iloc[:,5])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


class SvmTrain(object):
    def __init__(self, X, y, C, max_iters, tol, kernel, seta, gamma, degree):
        self.X = X
        self.y = y
        self.C = C
        self.max_iters = max_iters
        self.tol = tol
        self.n = len(self.X)
        self.kernel = kernel
        self.seta = seta
        self.gamma = gamma
        self.degree = degree
        
    def random_idx(self, i):
        j = i
        while j == i:
            j = np.random.randint(0, self.n)
        return j
    
    def find_bounds(self, a, i, j):
        if self.y[i] == self.y[j]:
            L = max(0, a[i] + a[j] - self.C)
            H = min(a[i] + a[j], self.C)
        else:
            L = max(0, a[j] - a[i])
            H = min(self.C, self.C + a[j] - a[j])
        return L, H
    
    def clip(self, a, L, H):
        if a > H:
            a = H
        elif a < L:
            a = L
        else:
            a = a
        return a
    
    def linear_kernel(self, X1, X2):
        res = X1.dot(X2.T)
        return res
    
    def poly_kernel(self, X1, X2):
        res = (self.seta + self.gamma*X1.dot(X2.T))**self.degree
        return res
    
    def rbf_kernel(self, X1, X2):
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        res = np.exp(-self.gamma * dist.cdist(X1, X2)**2)
        return res
    
    def train(self):
        if self.kernel == "linear":
            K = self.linear_kernel(self.X, self.X)
            kernel_func = self.linear_kernel
        elif self.kernel == "poly":
            assert isinstance(self.seta, int) and self.seta >= 0, "seta should be given as int and >= 0"
            assert isinstance(self.gamma, float) and self.gamma > 0, "gamma should be given as float and > 0"
            assert isinstance(self.degree, int) and self.degree >= 2, "degree should be given as int and >= 2"
            K = self.poly_kernel(self.X, self.X)
            kernel_func = self.poly_kernel
        elif self.kernel == "rbf":
            assert isinstance(self.gamma, float) and self.gamma > 0, "gamma should be given as float and > 0"
            K = self.rbf_kernel(self.X, self.X)
            kernel_func = self.rbf_kernel
        else:
            raise ValueError("kernel should be linear, poly or rbf")
        
        iters = 0
        a = np.zeros((1, self.n))[0]
        b = 0
        while iters < self.max_iters:
            iters += 1
            a_prev = copy.deepcopy(a)
            for i in range(self.n):
                Ki = kernel_func(self.X, self.X[i])
                ui = (a*self.y).T.dot(Ki) + b
                Ei = ui - self.y[i]
                if self.y[i]*Ei >= 1 and a[i] > 0 or self.y[i]*Ei <= 1 and a[i] < self.C or self.y[i]*Ei == 1 and a[i] == 0 or self.y[i]*Ei == 1 and a[i] == self.C:
                    # Pick random i
                    j = self.random_idx(i)
                    # Error for i
                    Kj = kernel_func(self.X, self.X[j])
                    uj = (a*self.y).T.dot(Kj) + b # y_hat
                    Ej = uj - self.y[j]
                    # 更新上下限
                    L, H = self.find_bounds(a, i, j)
                    # 计算eta
                    eta = K[i,i] + K[j,j] - 2*K[i,j]
#                    if eta <= 0: continue
                    # Save old alphas
                    ai_old, aj_old = a[i], a[j]
                    # Update alpha
                    a[j] = aj_old + self.y[j]*(Ei-Ej)/eta
                    a[j] = self.clip(a[j], L, H)
                    a[i] = ai_old + self.y[j]/self.y[i]*(aj_old-a[j])
                    # Find intercept
                    b1 = b - Ei - self.y[i]*(a[i]-ai_old)*K[i,i] - self.y[j]*(a[j]-aj_old)*K[i,j]
                    b2 = b - Ej - self.y[i]*(a[i]-ai_old)*K[i,j] - self.y[j]*(a[j]-aj_old)*K[j,j]
                    if 0 < a[i] < self.C:
                        b = b1
                    elif 0 < a[j] < self.C:
                        b = b2
                    else:
                        b = 0.5 * (b1 + b2)
                else:
                    continue
            # Check convergence
            diff = np.sqrt(np.sum((a-a_prev)**2))
            if diff < self.tol:
                break
            return a, b

obj = SvmTrain(X_train, y_train, C=0.6, max_iters=400, tol=0.001, kernel="linear", seta=None, gamma=None, degree=None)
obj = SvmTrain(X_train, y_train, C=0.6, max_iters=400, tol=0.001, kernel="poly", seta=1, gamma=0.01, degree=2)
obj = SvmTrain(X_train, y_train, C=0.6, max_iters=400, tol=0.001, kernel="rbf", seta=None, gamma=0.01, degree=None)
obj.train()

