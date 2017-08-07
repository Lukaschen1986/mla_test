# -*- coding: utf-8 -*-
# http://scikit-learn.org/stable/modules/neural_networks_supervised.html
from __future__ import division
import os
os.getcwd()
import numpy as np
import pandas as pd
import random as rd
from sklearn import datasets
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

unit_vec = lambda x: x/np.sqrt(np.sum(x**2))

class BpnnTrain(object):
    def __init__(self, X, y, layer_num, neuron_num, outNum, yita, lam, iter_num):
        self.X = X
        self.y = y
        self.layer_num = layer_num
        self.neuron_num = neuron_num
        self.outNum = outNum
        self.yita = yita
        self.lam = lam
        self.iter_num = iter_num
        self.n = len(self.X) # 样本个数
        self.p = self.X.shape[1] # 样本维度
    
    def layer_func(self):
        w_res = []; b_res = [] # 初始化w, b
        # 计算第一层w,b初始值
        w_begin = np.random.normal(0, 0.01, self.p*self.neuron_num).reshape(self.p,self.neuron_num)
        w_res.append(w_begin)
        b_begin = np.zeros((1, self.neuron_num))
        b_res.append(b_begin)
        # 计算中间层w,b初始值
        inter_num = self.layer_num-2 # 计算中间层个数
        for i in range(inter_num):
            w_inter = np.random.normal(0, 0.01, self.neuron_num*self.neuron_num).\
            reshape(self.neuron_num,self.neuron_num)
            b_inter = np.zeros((1, self.neuron_num))
            w_res.append(w_inter)
            b_res.append(b_inter)
            # 计算最后一层w,b初始值
        w_end = np.random.normal(0, 0.01, self.neuron_num*self.outNum).\
        reshape(self.neuron_num,self.outNum)
        w_res.append(w_end)
        b_end = np.zeros((1, self.outNum))
        b_res.append(b_end)
        return w_res, b_res
    
    def forward_func(self, w, b):
        a_res = []; a = self.X
        a_res.append(a)
        for i in range(len(w)-1):
            z = a.dot(w[i]) + b[i]
            a = np.tanh(z)
            a_res.append(a)
        z_end = a.dot(w[-1]) + b[-1]
        a_end = np.exp(z_end)
        prob = a_end/np.sum(a_end, axis=1, keepdims=True)
        return prob, a_res
    
    def loss_func(self, prob, w):
        # 计算损失函数值
        lossFunc = -np.sum(np.log(prob[range(self.n), self.y]))/self.n
        for i in range(len(w)):
            lossFunc += self.lam/2*np.sum(w[i]**2)
        return lossFunc
    
    def backward_func(self, prob, a, w):
        dw_res = []; db_res = []
        delta = prob
        delta[range(self.n), self.y] -= 1 # 对输出层求导
        for i in range(len(w))[::-1]: # 逐层求导，注意是倒序
            dw = a[i].T.dot(delta) + self.lam*w[i]
            db = np.sum(delta, axis=0, keepdims=True)
            dw_res.append(dw)
            db_res.append(db)
            delta = delta.dot(w[i].T) * (1-a[i]**2)
        # 把倒序转成正序 
        dw_res = dw_res[::-1]
        db_res = db_res[::-1]
        return dw_res, db_res

    def grad_descent(self, dw, db, w, b):
        for i in range(len(w)):
            w[i] -= self.yita*unit_vec(dw[i])
            b[i] -= self.yita*unit_vec(db[i])
        return w, b

    def iter_func(self):
        w, b = self.layer_func()
        lossFunc_res = []
        for i in range(self.iter_num):
            prob, a = self.forward_func(w, b)
            lossFunc = self.loss_func(prob, w)
            lossFunc_res.append(lossFunc)
            if lossFunc_res[i] > lossFunc_res[i-1]: break
            # 反向逐层求导
            dw, db = self.backward_func(prob, a, w)
            # 梯度下降更新w,b
            w, b = self.grad_descent(dw, db, w, b)
        return w, b, lossFunc_res
    
obj_train = BpnnTrain(X=X_train, y=y_train, layer_num=4, neuron_num=60, outNum=10, yita=0.001, lam=0.001, iter_num=10000)
w, b, lossFunc_res = obj_train.iter_func()
plt.plot(lossFunc_res)
min(lossFunc_res)
np.argmin(lossFunc_res)

class BpnnTest(BpnnTrain):
    def __init__(self, w, b, X, y):
        self.w = w
        self.b = b
        self.X = X
        self.y = y
    
    def test_func(self):
        prob, a = BpnnTrain.forward_func(self, w, b)
        y_hat = np.argmax(prob, axis=1)
        accu = np.sum(y_hat == self.y)/len(self.y)
        return y_hat, accu

obj_test = BpnnTest(w, b, X_test, y_test)
y_hat, accu = obj_test.test_func()

