# -*- coding: utf-8 -*-
import os
os.getcwd()
from scipy import linalg
import numpy as np
import pandas as pd
import scipy.spatial.distance as dist
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv("housing.csv")
x0 = np.ones((len(df),1))
x = df.iloc[:, 0:3].values
x = np.concatenate((x0, x), axis=1)
y = df.iloc[:, 3].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=0)

def linear_reg(x_train, y_train, x_test, y_test):
    # 计算 w
    xTx = x_train.T.dot(x_train)
    xTy = x_train.T.dot(y_train)
    assert linalg.det(xTx) != 0.0, "det should not be ZERO"
    w = xTy.dot(linalg.inv(xTx))
    # 预测
    y_hat = x_test.dot(w)
    # 评估
    rmse = np.sqrt(np.sum((y_test-y_hat)**2) / len(y_test))
    return y_hat, rmse
y_hat, rmse = linear_reg(x_train, y_train, x_test, y_test) # 98795


def locally_weighted_linear_reg(x_train, y_train, x_test, y_test, k):
    # 高斯权重
    get_sita = lambda x1, x2, k: np.exp(dist.cdist(x1, x2) / (-2.0*k**2))
    # 初始化 y_hat_res
    y_hat_res = []
    for i in range(len(x_test)):
        #i = 0
        sample = np.atleast_2d(x_test[i])
        # 计算每个预测样本距离训练样本的高斯权重 sita
        sita = get_sita(sample, x_train, k) 
        sita = np.diag(sita[0]) # 对角化
        # 计算每个预测样本的 w
        xTsx = x_train.T.dot(sita).dot(x_train)
        xTsy = x_train.T.dot(sita).dot(y_train)
        assert linalg.det(xTsx) != 0.0, "det should not be ZERO"
        w = xTsy.dot(linalg.inv(xTsx))
        # 预测
        y_hat = sample.dot(w)[0]
        y_hat_res.append(y_hat)
    y_hat_res = np.array(y_hat_res)
    # 评估
    rmse = np.sqrt(np.sum((y_test-y_hat_res)**2) / len(y_test))
    return y_hat_res, rmse
y_hat, rmse = locally_weighted_linear_reg(x_train, y_train, x_test, y_test, k=1)
