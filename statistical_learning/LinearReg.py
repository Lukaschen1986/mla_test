# -*- coding: utf-8 -*-
import os
import copy
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (LinearRegression, Ridge)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class LinearReg(object):
    def __init__(self, lamb):
        self.lamb = lamb
    
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
    
    def get_loss(self, x_train_scale, y_train, w):
        '''计算损失函数值'''
        y_hat = x_train_scale.dot(w)
        loss = 0.5 * np.mean((y_train - y_hat)**2) + 0.5 * self.lamb * np.mean(w**2)
        return loss
    
    def fit(self, x_train_scale, y_train):
        '''模型训练, lambda = 0即为普通线性回归'''
        A = x_train_scale.T.dot(x_train_scale)
        M = np.eye(A.shape[0]) * self.lamb
        w = np.linalg.inv(A + M).dot(x_train_scale.T).dot(y_train)
        return w
    
    def predict(self, x_test_scale, w):
        '''模型预测'''
        y_pred = x_test_scale.dot(w)
        return y_pred
        
        
if __name__ == "__main__":
    file_path = os.getcwd()
    dataSet = pd.read_csv(file_path + "/swiss_linear.csv")
    df = dataSet[["Fertility", "Agriculture", "Catholic", "InfantMortality"]]        
    
    x = df.iloc[:, 1: ]
    y = df.iloc[:, 0]        
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)        
    
    # 手写模型
    model = LinearReg(lamb=0.5)
    mu, std = model.z_scale(x_train)
    x_train_scale, x_test_scale = model.data_transform(mu, std, x_train, x_test)
    w = model.fit(x_train_scale, y_train)
    loss_train = model.get_loss(x_train_scale, y_train, w)
    y_pred = model.predict(x_test_scale, w)
    rmse = np.mean((y_test - y_pred)**2)
    print(f"LinearReg RMSE: {rmse}")
    
    # sklearn
    scale = StandardScaler(with_mean=True, with_std=True)
    scale.fit(x_train)
    x_train_scale = scale.transform(x_train)
    x_test_scale = scale.transform(x_test)
    
#    reg = LinearRegression(fit_intercept=True)
    reg = Ridge(alpha=0.5, fit_intercept=True)
    reg.fit(x_train_scale, y_train)
    w = reg.coef_
    b = reg.intercept_    
    y_pred = reg.predict(x_test_scale)
    rmse = np.mean((y_test - y_pred)**2)
    print(f"sklearn RMSE: {rmse}")
    
    # 画图
#    plt.figure(figsize=(8,6))
#    n_alphas = 20
#    alphas = np.logspace(-1,4,num=n_alphas)
#    coefs = []
#    for a in alphas:
#        ridge = Ridge(alpha=a, fit_intercept=False)
#        ridge.fit(X, y)
#        coefs.append(ridge.coef_[0])
#    ax = plt.gca()
#    ax.plot(alphas, coefs)
#    ax.set_xscale('log')
#    handles, labels = ax.get_legend_handles_labels()
#    plt.legend(labels=df.columns[1:-1])
#    plt.xlabel('alpha')
#    plt.ylabel('weights')
#    plt.axis('tight')
#    plt.show()
    
    # statsmodels
#    import statsmodels.api as sm #方法一
#    import statsmodels.formula.api as smf #方法二
#    est = sm.OLS(y, sm.add_constant(X)).fit() #方法一
#    est = smf.ols(formula='sales ~ TV + radio', data=df).fit() #方法二
#    y_pred = est.predict(X)
#    print(est.summary()) #回归结果
#    print(est.params) #系数
