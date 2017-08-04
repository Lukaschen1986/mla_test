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

digits = datasets.load_digits()
X = digits.data
y = digits.target

def data_split(data, prob):
    idx_train = rd.sample(range(len(data)), int(len(data)*prob))
    idx_test = [i for i in range(len(data)) if i not in idx_train]
    return idx_train, idx_test

idx_train, idx_test = data_split(X, 0.7)
X_train = X[idx_train]
y_train = y[idx_train]
X_test = X[idx_test]
y_test = y[idx_test]

def bpnn_train(X, y, hideNum_1=70, hideNum_2=60, hideNum_3=50, yita=0.001, lam=0.001, iterNum=10000):
    unit_vec = lambda x: x/np.sqrt(np.sum(x**2))
    # 定义超参
    n = len(X) # 样本个数
    p = X.shape[1] # 样本维度
    hideNum_1 = hideNum_1 # 隐藏神经元个数
    hideNum_2 = hideNum_2
    hideNum_3 = hideNum_3
    outNum = len(set(y)) # 输出层维度，等于y的水平个数
    yita = yita # 学习率
    lam = lam # 正则系数
    # 定义参数初始值
    np.random.seed(0)
#    w1 = np.random.randn(p, hideNum) / np.sqrt(p)
#    w2 = np.random.randn(hideNum, hideNum) / np.sqrt(hideNum)
#    w3 = np.random.randn(hideNum, outNum) / np.sqrt(hideNum)
    w1 = np.random.normal(0, 0.01, p*hideNum_1).reshape(p,hideNum_1)
    w2 = np.random.normal(0, 0.01, hideNum_1*hideNum_2).reshape(hideNum_1,hideNum_2)
    w3 = np.random.normal(0, 0.01, hideNum_2*hideNum_3).reshape(hideNum_2,hideNum_3)
    w4 = np.random.normal(0, 0.01, hideNum_3*outNum).reshape(hideNum_3,outNum)
    b1 = np.zeros((1, hideNum_1))
    b2 = np.zeros((1, hideNum_2))
    b3 = np.zeros((1, hideNum_3))
    b4 = np.zeros((1, outNum))
    lossFunc_res = []
    # 循环训练
    for i in range(iterNum):
        # 计算各输出
        z1 = X.dot(w1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(w2) + b2
        a2 = np.tanh(z2)
        z3 = a2.dot(w3) + b3
        a3 = np.tanh(z3)
        z4 = a3.dot(w4) + b4
        a4 = np.exp(z4)
        prob = a4/np.sum(a4, axis=1, keepdims=True)
        # 计算损失函数值，并判定是否跳出循环
        lossFunc = -np.sum(np.log(prob[range(n), y]))/n
        lossFunc += lam/2*np.sum(w1**2) + lam/2*np.sum(w2**2) + lam/2*np.sum(w3**2) + lam/2*np.sum(w4**2)
        lossFunc_res.append(lossFunc)
        if lossFunc_res[i] > lossFunc_res[i-1]: break
        # 判定完随即储存当前最优参数
        parameter = {"w1":w1, "b1":b1,
                     "w2":w2, "b2":b2,
                     "w3":w3, "b3":b3,
                     "w4":w4, "b4":b4}
        # 反向逐层求导
        delta4 = prob
        delta4[range(n), y] -= 1
        dw4 = a3.T.dot(delta4) + lam*w4
        db4 = np.sum(delta4, axis=0, keepdims=True)
                
        delta3 = delta4.dot(w4.T) * (1-a3**2)
        dw3 = a2.T.dot(delta3) + lam*w3
        db3 = np.sum(delta3, axis=0, keepdims=True)
        
        delta2 = delta3.dot(w3.T) * (1-a2**2)
        dw2 = a1.T.dot(delta2) + lam*w2
        db2 = np.sum(delta2, axis=0, keepdims=True)
        
        delta1 = delta2.dot(w2.T) * (1-a1**2)
        dw1 = X.T.dot(delta1) + lam*w1
        db1 = np.sum(delta1, axis=0, keepdims=True)
        # 梯度下降
        w4 -= yita*unit_vec(dw4)
        b4 -= yita*unit_vec(db4)
        w3 -= yita*unit_vec(dw3)
        b3 -= yita*unit_vec(db3)
        w2 -= yita*unit_vec(dw2)
        b2 -= yita*unit_vec(db2)
        w1 -= yita*unit_vec(dw1)
        b1 -= yita*unit_vec(db1)
    return parameter, lossFunc_res

parameter, lossFunc_res = bpnn_train(X_train, y_train, \
                                     hideNum_1=70, hideNum_2=70, hideNum_3=70, \
                                     yita=0.001, lam=0.001, iterNum=10000)
plt.plot(lossFunc_res); min(lossFunc_res); np.argmin(lossFunc_res)

def bpnn_predict(parameter, X, y):
    w1 = parameter["w1"]; b1 = parameter["b1"]
    w2 = parameter["w2"]; b2 = parameter["b2"]
    w3 = parameter["w3"]; b3 = parameter["b3"]
    w4 = parameter["w4"]; b4 = parameter["b4"]
    z1 = X.dot(w1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(w2) + b2
    a2 = np.tanh(z2)
    z3 = a2.dot(w3) + b3
    a3 = np.tanh(z3)
    z4 = a3.dot(w4) + b4
    a4 = np.exp(z4)
    prob = a4/np.sum(a4, axis=1, keepdims=True)
    y_hat = np.argmax(prob, axis=1)
    accu = np.sum(y_hat == y)/len(y)
    return y_hat, accu
y_hat, accu = bpnn_predict(parameter, X_test, y_test)
print(accu)

# MLPClassifier
clf = MLPClassifier(activation="tanh", hidden_layer_sizes=(70,70,70), random_state=1)
clf.fit(X_train, y_train)
y_hat_sk = clf.predict(X_test)
np.sum(y_hat_sk == y_test)/len(y_test)

# LR
clf = linear_model.LogisticRegressionCV()
clf.fit(X_train, y_train)
clf_pred = clf.predict(X_test)
np.sum(clf_pred == y_test)/len(y_test)
