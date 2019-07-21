# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron

class PerceptronModel(object):
    def __init__(self, w_init, b_init, lr, max_iter):
        self.w_init = w_init
        self.b_init = b_init
        self.lr = lr
        self.max_iter = max_iter
    
    def get_loss(self, x_train, y_train, w, b):
        wrong_sample = (y_train.reshape(-1,1) * (x_train.dot(w) + b) <= 0).reshape(-1)
        x_wrong = x_train[wrong_sample]
        y_wrong = y_train[wrong_sample]
        loss = -y_wrong.dot(x_wrong.dot(w) + b)
        return loss[0]
    
    def fit(self, x_train, y_train):
        # 计算初始损失值
        loss_init = self.get_loss(x_train, y_train, self.w_init, self.b_init)
        loss_res = []; loss_res.append(loss_init)
        # 保存参数
        w_res = []; w_res.append(self.w_init)
        b_res = []; b_res.append(self.b_init)
        # 迭代优化
        for step in range(self.max_iter):
            # 随机抽样一个样本
            idx = rd.sample(range(len(x_train)), 1)
            x_, y_ = x_train[idx], y_train[idx]
            # 当出现误判时进行梯度下降
            if y_ * (x_.dot(w_res[-1]) + b_res[-1]) <= 0:
                w = w_res[-1] + (self.lr*y_*x_).T
                b = b_res[-1] + self.lr*y_
            else:
                continue
            # 更新损失值，保存参数
            loss = self.get_loss(x_train, y_train, w, b)
            loss_res.append(loss); w_res.append(w); b_res.append(b)
        # 取损失值最小的参数为最优参数
        w_best = w_res[np.argmin(loss_res)]
        b_best = b_res[np.argmin(loss_res)]
        return loss_res, w_best, b_best
    
    def predict(self, x, w, b):
        y_pred = np.sign(x.dot(w) + b).reshape(-1)
        return y_pred
    
    def get_score(self, y_true, y_pred):
        score = sum(y_true == y_pred) / len(y_true)
        return score


if __name__ == "__main__":
    # 构造二分类数据集
    N = 500
    x1 = np.random.uniform(low=1, high=5, size=[N,2]) + np.random.randn(N, 2)*0.01
    y1 = np.tile(-1, N)
    
    x2 = np.random.uniform(low=5, high=10, size=[N,2]) + np.random.randn(N, 2)*0.01
    y2 = np.tile(1, N)
    
    x = np.concatenate([x1,x2], axis=0)
    y = np.concatenate([y1,y2])
    
    x, y = shuffle(x, y, random_state=0)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    
    # 参数初始化
    w_init = np.zeros([x.shape[1], 1]) + 0.001
    b_init = 0
    lr = 0.01
    max_iter = 5000
    
    # 运行自写算法
    model = PerceptronModel(w_init, b_init, lr, max_iter)
    loss_res, w_best, b_best = model.fit(x_train, y_train)
    
    y_pred = model.predict(x_test, w_best, b_best)
    score = model.get_score(y_test, y_pred)
    print(f"PerceptronModel 预测准确率：{score}")
    
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(loss_res)
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.title("Loss")
    plt.show()

    fig, ax = plt.subplots(figsize=(8,6))
    x_axis = np.linspace(1, 10, 10)
    y_axis = -(w_best[0]*x_axis + b_best) / w_best[1]
    ax.plot(x_axis, y_axis, color="red")
    ax.scatter(x=x_test[:,0], y=x_test[:,1], c=y_test, label=y_test)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("PerceptronModel")
    plt.show()
    
    # 收敛性验证
    mod_res = []
    for i in range(len(x_train)):
        mod = np.sqrt(np.sum(x_train[i]**2))
        mod_res.append(mod)
    R = np.max(mod_res)
    w_aug = np.append(w_best, b_best)
    C = np.sqrt(np.sum(w_aug**2))
    r = np.min(y_train.reshape(-1,1) * (x_train.dot(w_best) + b_best))
    k = np.floor((R*C / r)**2)
    print(f"理论上，基于当前数据集，算法的收敛上界为{k}轮")
    
    # 运行sklearn算法
    clf = Perceptron(eta0=lr, fit_intercept=True, max_iter=max_iter, tol=0.001)
    clf.fit(x_train, y_train)
    
    w = clf.coef_[0]
    b = clf.intercept_
    y_pred = clf.predict(x_test)
    score = sum(y_test == y_pred) / len(y_test)
    print(f"PerceptronSklearn 预测准确率：{score}")
        
    fig, ax = plt.subplots(figsize=(8,6))
    x_axis = np.linspace(1, 10, 10)
    y_axis = -(w[0]*x_axis + b) / w[1]
    ax.plot(x_axis, y_axis, color="red")
    ax.scatter(x=x_test[:,0], y=x_test[:,1], c=y_test, label=y_test)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("PerceptronModel")
    plt.show()
