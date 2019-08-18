# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
#from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class LogitRegModel(object):
    def __init__(self, max_iter=5000, eta=0.01, alpha=0.5, beta=0.9):
        self.max_iter = max_iter
        self.eta = eta
        self.alpha = alpha
        self.beta = beta
    
    
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
        loss = np.mean(np.log(1.0 + np.exp(-x_train_scale.dot(w) * y_train)))
        return loss
    
    
    def get_derivative(self, x_train_scale, y_train, w, dv):
        '''计算梯度(含动量, beta = 0 则为原始梯度下降)'''
        fenzi = -y_train * x_train_scale
        fenmu = 1.0 + np.exp(x_train_scale.dot(w) * y_train)
        
        dw = np.mean(fenzi / fenmu, axis=0)
        dw = dw.reshape(-1, 1)
        
        dv = self.beta * dv + (1 - self.beta) * dw
        return dv
    
    
    def fit(self, x_train_scale, y_train):
        '''模型训练'''
        # 参数初始化
        w = np.zeros(x_train_scale.shape[1]) + 0.001
        w = w.reshape(-1, 1)
        dv = np.zeros_like(w)
        # 损失值保存列表
        loss_res = []
        # 迭代
        for epoch in range(self.max_iter):
            # 计算梯度
            dv = self.get_derivative(x_train_scale, y_train, w, dv)
            # 梯度下降
            w = w - self.eta * dv
            # 更新损失值
            loss = self.get_loss(x_train_scale, y_train, w)
            loss_res.append(loss)
        return w, loss_res

    
    def predict(self, x_test_scale, w):
        '''模型预测'''
        y_pred_probs = 1.0 / (1.0 + np.exp(-x_test_scale.dot(w)))
        y_pred = np.where(y_pred_probs > self.alpha, 1, -1)
        return y_pred_probs, y_pred
    
    
    def get_score(self, y_true, y_pred):
        '''模型评估'''
        score = sum(y_true == y_pred) / len(y_true)
        return score



class AdaBoostModel(LogitRegModel):
    def __init__(self, ada_iter, base_iter, eta, alpha, beta):
        self.ada_iter = ada_iter
        super(AdaBoostModel, self).__init__(base_iter, eta, alpha, beta)
    
    
    def update_weights(self, weights, y_train, a, y_pred):
        '''更新样本权重'''
        Z = np.sum(weights * np.exp(-y_train * a * y_pred))
        weights = weights * np.exp(-y_train * a * y_pred) / Z
        return weights
        
    
    def fit_adaboost(self, x_train_scale, y_train, x_test_scale, y_test):
        '''模型训练'''
        # 定义子模型
        base_model = LogitRegModel()
        # 样本权重初始化
        weights = 1.0 / x_train_scale.shape[0]
        weights = np.ones_like(y_train) * weights
        d_train_scale = weights * x_train_scale
        # 参数初始化
        a_res = []
        theta_res = []
        G_train = 0
        G_test = 0
        loss_train_res = []
        loss_test_res = []
        
        for epoch in range(self.ada_iter):
            # 根据定义好的基分类器训练模型 g1，并保存本轮模型参数
            theta, loss_res = base_model.fit(d_train_scale, y_train)
            theta_res.append(theta)
            # 计算本轮的 误分类率 和 alpha 最优值，并保存 alpha*
            y_pred_probs, y_pred = base_model.predict(d_train_scale, theta)
            e = weights[y_train != y_pred].sum()
            a = 0.5 * np.log((1.0 - e) / e)
            a_res.append(a)
            # 计算 累积截止到本轮 的 训练集损失 和 测试集损失
            _, y_pred_train = base_model.predict(x_train_scale, theta)
            _, y_pred_test = base_model.predict(x_test_scale, theta)
            
            G_train += a * y_pred_train
            G_test += a * y_pred_test
            
            loss_train = np.mean(np.exp(-y_train * G_train))
            loss_test = np.mean(np.exp(-y_test * G_test))
            loss_train_res.append(loss_train)
            loss_test_res.append(loss_test)
            # 计算下一轮的样本权重分布 w
            weights = self.update_weights(weights, y_train, a, y_pred)
            # 更新样本权重赋值
            d_train_scale = weights * x_train_scale
        return a_res, theta_res, loss_train_res, loss_test_res
            
    
    def choose_best(self, loss_test_res, a_res, theta_res):
        '''确定最优分类器个数'''
        best_iter = np.argmin(loss_test_res)
        print(f"最优迭代次数（基于测试集）：{best_iter}")
        a_best = a_res[0: best_iter + 1]
        theta_best = theta_res[0: best_iter + 1]
        return a_best, theta_best
    
    
    def predict(self, x_test_scale, a_res, theta_res):
        '''模型预测'''
        # 定义子模型
        base_model = LogitRegModel()
        # 参数初始化
        G_test = 0
        # 综合子模型预测值
        for i in range(len(a_res)):
            _, y_pred_test = base_model.predict(x_test_scale, theta_res[i])
            G_test += a_res[i] * y_pred_test
            
        y_pred = np.sign(G_test)
        return y_pred
    
    
    def get_score(self, y_true, y_pred):
        '''模型评估'''
        score = sum(y_true == y_pred) / len(y_true)
        return score



if __name__ == "__main__":
    # 构造二分类数据集
    N = 200; n = 4
    x1 = np.random.uniform(low=1, high=5, size=[N, n]) + np.random.randn(N, n)
    y1 = np.tile(-1, N)
    
    x2 = np.random.uniform(low=4, high=10, size=[N, n]) + np.random.randn(N, n)
    y2 = np.tile(1, N)
    
    x = np.concatenate([x1, x2], axis=0)
    y = np.concatenate([y1, y2]).reshape(-1, 1)
    
    x, y = shuffle(x, y, random_state=0)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    
    # 手写模型
    model = AdaBoostModel(ada_iter=10, base_iter=100, eta=0.01, alpha=0.5, beta=0.9)
    mu, std = model.z_scale(x_train)
    x_train_scale, x_test_scale = model.data_transform(mu, std, x_train, x_test)
    
    a_res, theta_res, loss_train_res, loss_test_res = model.fit_adaboost(x_train_scale, y_train,
                                                                         x_test_scale, y_test)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    pd.DataFrame({"loss_train": loss_train_res, "loss_test": loss_test_res}).plot()
    plt.xlabel("a")
    plt.ylabel("loss")
    plt.title("AdaBoostModel Loss")
    plt.show()

    a_best, theta_best = model.choose_best(loss_test_res, a_res, theta_res)
    
    y_pred = model.predict(x_test_scale, a_best, theta_best)
    score = model.get_score(y_test, y_pred)
    print(f"AdaBoostModel 预测准确率：{score}")
    
    # sklearn
    scale = StandardScaler(with_mean=True, with_std=True)
    scale.fit(x_train)
    x_train_scale = scale.transform(x_train)
    x_test_scale = scale.transform(x_test)
    
    clf = AdaBoostClassifier(n_estimators=50)
    clf.fit(x_train_scale, y_train)
    
    y_pred = clf.predict(x_test_scale).reshape(-1, 1)
    score = sum(y_test == y_pred) / len(y_test)
    print(f"Sklearn 预测准确率：{score}")
