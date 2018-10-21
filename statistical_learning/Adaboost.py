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
import matplotlib.pyplot as plt


class LogisticRegression(object):
    def __init__(self, eta, max_iter, tol, threshold):
        self.eta = eta
        self.max_iter = max_iter
        self.tol = tol
        self.threshold = threshold
        
    def get_normal_scale(self, x_train, x_test):
        mean = np.mean(x_train, axis=0)
        std = np.std(x_train, axis=0)
        x_train_transform = (x_train - mean) / std
        x_test_transform = (x_test - mean) / std
        return x_train_transform, x_test_transform
    
    def add_intercept(self, x):
        intercept = np.ones((len(x),1))
        x = np.concatenate((intercept, x), axis=1)
        return x
    
    def fit(self, x_train_transform, y_train):
        unit_vector = lambda x: x / np.sqrt(np.sum(x**2))
        x = x_train_transform
        y = np.expand_dims(y_train, axis=1)
        nrow, ncol = x.shape
        w = np.zeros((ncol, 1)) + 0.001
        
        lost_log = []
        for i in range(self.max_iter):
            g = (-y*x).T.dot(1/(1+np.exp(y*x.dot(w)))) / nrow
            g_unit = unit_vector(g) 
            w -= self.eta * g_unit
            lost = np.sum(np.log(1 + np.exp(-y * x.dot(w)))) / nrow
            lost_log.append(lost)
            # break condition
            if (i >= 1) and (np.abs(lost_log[i]-lost_log[i-1]) <= self.tol):
                break
        return lost_log, w
    
    def predict(self, x_test_transform, w):
        y_proba = 1 / (1+np.exp(-x_test_transform.dot(w)))
        y_hat = np.where(y_proba > self.threshold, 1, -1)
        y_hat = y_hat.reshape(len(y_hat),)
        return y_hat
    
    def metric(self, y_true, y_predict):
        accu = np.sum(y_true == y_predict) / len(y_true)
        return accu
    

class Adaboost(LogisticRegression):
    def __init__(self, eta, max_iter, tol, threshold, steps):
        LogisticRegression.__init__(self, eta, max_iter, tol, threshold)
        self.steps = steps
    
    def init_weights(self, x):
        nrow = len(x)
        init_value = 1.0 / nrow
        weights = np.tile(init_value, nrow)
        return weights
    
    def update_epsilon(self, weights, y_train, y_hat):
        eps = np.sum(weights[y_train != y_hat])
        return eps
    
    def update_alpha(self, eps):
        alpha = 0.5 * np.log((1-eps) / eps)
        return alpha
    
    def update_weights(self, weights, y_train, alpha, y_hat):
        z = np.sum(weights * np.exp(-alpha*y_train*y_hat))
        weights = weights * np.exp(-alpha*y_train*y_hat) / z
        return weights
    
    def iter_func(self, x_train_transform):
        weights = self.init_weights(x_train_transform)
#        x_train_update = np.concatenate((x_train_transform, np.expand_dims(weights, axis=1)), axis=1)
        x_train_update = np.expand_dims(weights, axis=1) * x_train_transform
        nrow, ncol = x_train_update.shape
        w_res = np.zeros((ncol, self.steps))
        eps_res = []; alpha_res = []
        
        for i in range(self.steps):
            lost_log, w = self.fit(x_train_update, y_train) 
            w_res[:,i] = w.T
            y_hat = self.predict(x_train_update, w)
            # update_epsilon
            eps = self.update_epsilon(weights, y_train, y_hat)
            eps_res.append(eps)
            # update_alpha
            alpha = self.update_alpha(eps)
            alpha_res.append(alpha)
            if alpha == 0.0:
                self.steps = i
                w_res = w_res[:,0:i]
                break
            # update_weights
            weights = self.update_weights(weights, y_train, alpha, y_hat)
#            x_train_update = np.concatenate((x_train_transform, np.expand_dims(weights, axis=1)), axis=1)
            x_train_update = np.expand_dims(weights, axis=1) * x_train_transform
        return eps_res, alpha_res, w_res
    
    def predict_func(self, alpha_res, x_test_transform, w_res):
        y_pred = 0 
    
        for i in range(self.steps):
            y_proba = 1 / (1+np.exp(-x_test_transform.dot(w_res[:,i])))
            y_hat = np.where(y_proba > self.threshold, 1, -1)
            y_pred += alpha_res[i] * y_hat
        
        y_pred = np.sign(y_pred)
        return y_pred
    
#    def predict_func(self, alpha_res, x_test_transform, w_res):
#        weights = self.init_weights(x_test_transform)
##        x_test_update = np.concatenate((x_test_transform, np.expand_dims(weights, axis=1)), axis=1)
#        x_test_update = np.expand_dims(weights, axis=1) * x_test_transform
#        y_pred = 0 
#    
#        for i in range(self.steps):
#            y_proba = 1 / (1+np.exp(-x_test_update.dot(w_res[:,i])))
#            y_hat = np.where(y_proba > self.threshold, 1, -1)
#            y_pred += alpha_res[i] * y_hat
#        
#        y_pred = np.sign(y_pred)
#        return y_pred
    

if __name__ == "__main__":
    dataSet = pd.read_table("data_banknote_authentication.txt", sep=",", names=["x1","x2","x3","x4","y"])
    dataSet.loc[dataSet.y == 0, "y"] = -1
    dataSet = shuffle(dataSet)
    df_train, df_test = train_test_split(dataSet, test_size=0.2)
    
    x_train = df_train.iloc[:,0:4]
    y_train = df_train.iloc[:,4]
    x_test = df_test.iloc[:,0:4]
    y_test = df_test.iloc[:,4]
    
    LR = LogisticRegression(eta=0.01, max_iter=5000, tol=10**-8, threshold=0.5)
    x_train_transform, x_test_transform = LR.get_normal_scale(x_train, x_test)
    x_train_transform = LR.add_intercept(x_train_transform)
    x_test_transform = LR.add_intercept(x_test_transform)
    lost_log, w = LR.fit(x_train_transform, y_train)
    plt.plot(lost_log)
    y_hat = LR.predict(x_test_transform, w)
    confusion_matrix(y_test, y_hat)
    f1_score(y_test, y_hat)
    
    AB = Adaboost(eta=0.01, max_iter=1000, tol=10**-8, threshold=0.5, steps=200)
    x_train_transform, x_test_transform = AB.get_normal_scale(x_train, x_test)
    x_train_transform = AB.add_intercept(x_train_transform)
    x_test_transform = AB.add_intercept(x_test_transform)
    eps_res, alpha_res, w_res = AB.iter_func(x_train_transform)
    plt.plot(eps_res)
    plt.plot(alpha_res)
    y_pred = AB.predict_func(alpha_res, x_test_transform, w_res)
    confusion_matrix(y_test, y_pred)
    f1_score(y_test, y_pred)
    
