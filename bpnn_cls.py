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

class BPNetworkCls(object):
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
        
    unit_vec = lambda x: x/np.sqrt(np.sum(x**2))
    
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
    
obj = BPNetworkCls(X=X_train, y=y_train, layer_num=4, neuron_num=4, outNum=10, yita=0.001, lam=0.001, iter_num=1000)
w, b = obj.layer_func()
prob, a_res = obj.forward_func(w,b)


