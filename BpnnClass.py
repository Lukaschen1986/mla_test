# -*- coding: utf-8 -*-
from __future__ import division
import os
os.getcwd()
import numpy as np
import pandas as pd
import random as rd
from sklearn import datasets
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

digits = datasets.load_digits()
data = digits.data
target = digits.target
target = np.where(target <= 5, 0, 1)

#t = []
#for var in target:
#    if var <= 3:
#        val = 0
#    elif var > 3 and var <= 6:
#        val = 1
#    else:
#        val = 2
#    t.append(val)
#target = np.array(t)


#df = pd.read_csv("swiss2.csv", sep=",")
#data = np.array(df)[:,0:5]
#target = np.array(df)[:,5].astype(int)

X, X_predict, y, y_predict = train_test_split(data, target, test_size=0.2)


class BpnnTrain(object):
    def __init__(self, eta, lam, iters, neuNum, layerNum, outNum, active_in):
#        self.X = X
#        self.y = y
#        self.n = X.shape[0]
#        self.p = X.shape[1]
        self.eta = eta
        self.lam = lam
        self.iters = iters
        self.neuNum = neuNum
        self.layerNum = layerNum
        self.outNum = outNum
        self.active_in = active_in
#        self.active_out = active_out
    
    def sigmiod(self, z):
        res = 1.0 / (1.0 + np.exp(-z))
        return res
    
    def tanh(self, z):
        res = 2.0 / (1.0 + np.exp(-z)) - 1
        return res
    
    def relu(self, z):
        res = np.maximum(0, z)
        return res
    
    def leaky_relu(self, z):
        res = np.maximum(0.01*z, z)
        return res
    
    def softmax(self, z):
        res = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
        return res
    
    def tanh_dv(self, z):
        res = 1-(np.tanh(z))**2
        return res
    
    def relu_dv(self, z):
        res = np.where(z < 0, 0, 1)
        return res
    
    def leaky_relu_dv(self, z):
        res = np.where(z < 0, 0.01, 1)
        return res
    
    def unit_vec(self, x):
        res = x/np.sqrt(np.sum(x**2))
        return res
    
    def layer_func(self, X):
        assert self.layerNum >= 2, "layerNum must >= 2"
        assert isinstance(self.layerNum, int), "layerNum must as int"
        assert isinstance(self.neuNum, int), "neuNum must as int"
        # 创建w,b集合
        w_res = []; b_res = []
        p = X.shape[1]
        # 计算输入层w,b初始值
        w_1 = np.random.randn(p, self.neuNum) * 0.01
        w_res.append(w_1)
        b_1 = np.zeros((1, self.neuNum))
        b_res.append(b_1)
        # 计算中间层w,b初始值
        for i in range(self.layerNum-1):
            w = np.random.randn(self.neuNum, self.neuNum) * 0.01
            b = np.zeros((1, self.neuNum))
            w_res.append(w)
            b_res.append(b)
        # 计算输出层w,b初始值
        w_end = np.random.randn(self.neuNum, self.outNum) * 0.01
        b_end = np.zeros((1, self.outNum))
        w_res.append(w_end)
        b_res.append(b_end)
        return w_res, b_res
    
    def forward_func(self, X, w, b):
        # 定义中间层激活函数
        if self.active_in == "tanh":
            activeIn = self.tanh
        elif self.active_in == "relu":
            activeIn = self.relu
        elif self.active_in == "leaky_relu":
            activeIn = self.leaky_relu
        else:
            ValueError("active_in must in ('tanh','relu','leaky_relu')")
        # 定义输出层激活函数
#        if self.active_out == "sigmoid":
#            activeOut = self.sigmiod
#        elif self.active_out == "softmax":
#            activeOut = self.softmax
#        else:
#            ValueError("active_out must in ('sigmoid','softmax')")
        # forward
        a = X
        a_res = []; z_res = []
        a_res.append(a); z_res.append(0)
        for i in range(len(w)-1):
            # i = 1; w = w_res; b = b_res; active_in = tanh; active_out = softmax
            z = a.dot(w[i]) + b[i] # z1
            a = activeIn(z)
            z_res.append(z)
            a_res.append(a)
        z_end = a.dot(w[-1]) + b[-1]
        a_end = self.softmax(z_end)
        return a_res, z_res, a_end
    
    def loss_func(self, X, y, a_end, w):
        n = X.shape[0]
        y_hat = a_end[range(n), y]
        loss = -np.sum(np.log(y_hat))/n
#        if self.active_out == "sigmoid":
#            loss = -np.sum(y*np.log(y_hat)+(1-y)*np.log(1-y_hat))/n
#        elif self.active_out == "softmax":
#            loss = -np.sum(np.log(y_hat))/n
#        else:
#            ValueError("active_out must in ('sigmoid','softmax')")
        for i in range(len(w)):
            loss += self.lam/2*np.sum(w[i]**2)
        return loss
    
    def backward_func(self, X, y, a, z, a_end, w):
        dw_res = []; db_res = []
        n = X.shape[0]
        # 输出层求导
        delta = a_end
        delta[range(n), y] -= 1
#        if self.active_out == "sigmoid":
#            delta = (a_end-y[:,np.newaxis])/n
#        elif self.active_out == "softmax":
#            delta = a_end
#            delta[range(n), y] -= 1
#        else:
#            ValueError("active_out must in ('sigmoid', 'softmax')")
        # 隐藏层求导函数
        if self.active_in == "tanh":
            activeDv = self.tanh_dv
        elif self.active_in == "relu":
            activeDv = self.relu_dv
        elif self.active_in == "leaky_relu":
            activeDv = self.leaky_relu_dv
        else:
            ValueError("active_in must in (tanh, relu, leaky_relu)")
        # 逐层求导，注意是倒序    
        for i in range(len(w))[::-1]:
            dw = a[i].T.dot(delta) + self.lam*w[i]
            db = np.sum(delta, axis=0, keepdims=True)
            dw_res.append(dw)
            db_res.append(db)
            delta = delta.dot(w[i].T) * activeDv(z[i])
        # 把倒序转成正序 
        dw_res = dw_res[::-1]
        db_res = db_res[::-1]
        return dw_res, db_res
    
    def grad_descent(self, dw, db, w, b):
        for i in range(len(w)):
            w[i] -= self.eta * self.unit_vec(dw[i])
            b[i] -= self.eta * self.unit_vec(db[i])
        return w, b
    
    def train(self, X, y):
        w, b = self.layer_func(X)
        loss_res = []
        for i in range(self.iters):
            a, z, a_end = self.forward_func(X, w, b)
            loss = self.loss_func(X, y, a_end, w)
            loss_res.append(loss)
            if loss_res[i] > loss_res[i-1]: 
                break
            # 反向逐层求导
            dw, db = self.backward_func(X, y, a, z, a_end, w)
            # 梯度下降更新w,b
            w, b = self.grad_descent(dw, db, w, b)
        return w, b, loss_res
    
obj_train = BpnnTrain(eta=0.001, lam=0.001, iters=10000, neuNum=60, layerNum=3, outNum=10, active_in="tanh")
obj_train = BpnnTrain(eta=0.001, lam=0.001, iters=10000, neuNum=60, layerNum=3, outNum=10, active_in="relu")
obj_train = BpnnTrain(eta=0.001, lam=0.001, iters=10000, neuNum=60, layerNum=3, outNum=10, active_in="leaky_relu")
obj_train = BpnnTrain(eta=0.001, lam=0.001, iters=10000, neuNum=70, layerNum=3, outNum=2, active_in="tanh")
obj_train = BpnnTrain(eta=0.001, lam=0.001, iters=10000, neuNum=60, layerNum=3, outNum=2, active_in="tanh")

w, b, loss_res = obj_train.train(X, y)
plt.plot(loss_res)
min(loss_res)
np.argmin(loss_res)

class BpnnPredict(BpnnTrain):
    def __init__(self, w, b, active_in):
#        BpnnTrain.__init__(self, active_in, active_out)
        self.w = w
        self.b = b
        self.active_in = active_in
#        self.active_out = active_out
    
    def predict(self, X_predict):
        a, z, a_end = BpnnTrain.forward_func(self, X_predict, w, b)
        y_pred = np.argmax(a_end, axis=1)
        return y_pred, a_end
    
obj_predict = BpnnPredict(w, b, active_in="tanh")
obj_predict = BpnnPredict(w, b, active_in="relu")
obj_predict = BpnnPredict(w, b, active_in="leaky_relu")
obj_predict = BpnnPredict(w, b, active_in="tanh")
obj_predict = BpnnPredict(w, b, active_in="tanh")

y_pred, a_end = obj_predict.predict(X_predict)
np.sum(y_pred == y_predict)/len(y_predict)
