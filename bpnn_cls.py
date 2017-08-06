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

def layer_func(p, outNum, layer_num, neuron_num):
    w_res = []; b_res = [] # 初始化w, b
    # 计算第一层w,b初始值
    w_begin = np.random.normal(0, 0.01, p*neuron_num).reshape(p,neuron_num)
    w_res.append(w_begin)
    b_begin = np.zeros((1, neuron_num))
    b_res.append(b_begin)
    # 计算中间层w,b初始值
    inter_num = layer_num-2 # 计算中间层个数
    for i in range(inter_num):
        w_inter = np.random.normal(0, 0.01, neuron_num*neuron_num).reshape(neuron_num,neuron_num)
        b_inter = np.zeros((1, neuron_num))
        w_res.append(w_inter)
        b_res.append(b_inter)
    # 计算最后一层w,b初始值
    w_end = np.random.normal(0, 0.01, neuron_num*outNum).reshape(neuron_num,outNum)
    w_res.append(w_end)
    b_end = np.zeros((1, outNum))
    b_res.append(b_end)
    return w_res, b_res

def forward_func(X_input, w, b):
    a_res = []; a = X_input
    a_res.append(a)
    for i in range(len(w)-1):
        z = a.dot(w[i]) + b[i]
        a = np.tanh(z)
        a_res.append(a)
    z_end = a.dot(w[-1]) + b[-1]
    a_end = np.exp(z_end)
    prob = a_end/np.sum(a_end, axis=1, keepdims=True)
    return prob, a_res

def loss_func(prob, n, y, lam, w):
    # 计算损失函数值
    lossFunc = -np.sum(np.log(prob[range(n), y]))/n
    for i in range(len(w)):
        lossFunc += lam/2*np.sum(w[i]**2)
    return lossFunc

def backward_func(prob, a, n, y, lam, w):
    dw_res = []; db_res = []
    delta = prob
    delta[range(n), y] -= 1 # 输出层求导
    for i in range(len(w))[::-1]: # 逐层求导，注意是倒序
        dw = a[i].T.dot(delta) + lam*w[i]
        db = np.sum(delta, axis=0, keepdims=True)
        dw_res.append(dw)
        db_res.append(db)
        delta = delta.dot(w[i].T) * (1-a[i]**2)
    # 把倒序转成正序 
    dw_res = dw_res[::-1]
    db_res = db_res[::-1]
    return dw_res, db_res

def grad_descent(dw, db, w, b, yita):
    for i in range(len(w)):
        w[i] -= yita*unit_vec(dw[i])
        b[i] -= yita*unit_vec(db[i])
    return w, b
