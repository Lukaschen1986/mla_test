# -*- coding: utf-8 -*-
import os
import copy
import numpy as np
import pandas as pd
import scipy.spatial.distance as dist
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (LinearRegression, Ridge)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class LocallyWeightedReg(object):
    def __init__(self, k):
        self.k = k
    
    def z_scale(self, x_train):
        '''z标准化，在动用距离度量的算法中，必须先进行标准化以消除数据量纲的影响'''
        mu = np.mean(x_train, axis=0)
        std = np.std(x_train, axis=0)
        return mu, std
    
    def data_transform(self, mu, std, x_train, x_test):
        '''
        数据变换
        1、执行标准化操作
        2、插入截距项
        '''
        x_train_scale = (x_train - mu) / std
        x_test_scale = (x_test - mu) / std
        
        intercept_train = np.ones(x_train_scale.shape[0]).reshape(-1, 1)
        intercept_test = np.ones(x_test_scale.shape[0]).reshape(-1, 1)
        
        x_train_scale = np.concatenate([intercept_train, x_train_scale], axis=1)
        x_test_scale = np.concatenate([intercept_test, x_test_scale], axis=1)
        return x_train_scale, x_test_scale
    
    def predict(self, x_train_scale, x_test_scale, y_train, y_test):
        '''模型预测'''
        # theta矩阵：预测集和训练集元素距离矩阵
        Theta = np.exp(-0.5 * dist.cdist(x_test_scale, x_train_scale)**2 / self.k**2)
        # 预测值保存
        y_pred = np.array([])
        # 对每个预测点进行循环
        for i in range(len(Theta)):
            theta_sample = np.eye(x_train_scale.shape[0]) * Theta[i, :] # 每个预测点与训练集的距离对角阵
            A = x_train_scale.T.dot(theta_sample).dot(x_train_scale)
            w_sample = np.linalg.inv(A).dot(x_train_scale.T).dot(theta_sample).dot(y_train)
            print(f"sample: {i}; w: {w_sample}")
            y_pred_sample = x_test_scale[i, :].dot(w_sample)
            y_pred = np.append(y_pred, y_pred_sample)
        return y_pred

    
if __name__ == "__main__":
    file_path = os.getcwd()
    dataSet = pd.read_csv(file_path + "/swiss_linear.csv")
    df = dataSet[["Fertility", "Agriculture", "Catholic", "InfantMortality"]]        
    
    x = df.iloc[:, 1: ]
    y = df.iloc[:, 0]        
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)        
    
    # 手写模型
    model = LocallyWeightedReg(k=50)
    mu, std = model.z_scale(x_train)
    x_train_scale, x_test_scale = model.data_transform(mu, std, x_train, x_test)
    y_pred = model.predict(x_train_scale, x_test_scale, y_train, y_test)
    rmse = np.mean((y_test - y_pred)**2)
    print(f"LocallyWeightedReg RMSE: {rmse}")
