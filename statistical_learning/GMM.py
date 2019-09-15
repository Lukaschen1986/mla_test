# -*- coding: utf-8 -*-
import os
import copy
#import random as rd
import numpy as np
import pandas as pd
#from sklearn.utils import shuffle
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from sklearn.metrics import silhouette_score


class GMM(object):
    def __init__(self, z, tol, max_iter, init_params, verbose):
        self.z = z
        self.tol = tol
        self.max_iter = max_iter
        self.init_params = init_params
        self.verbose = verbose
        
        
    def fit(self, x):
        '''模型训练'''
        # 参数初始化
        N, n = x.shape
        Q = np.zeros([N, self.z])
        pdf = np.ones([N, self.z])
        P = np.random.uniform(low=0, high=1, size=(self.z, 1))
        
        if self.init_params == "kmeans":
            km = KMeans(n_clusters=self.z, init="k-means++")
            km.fit(x)
            mu = km.cluster_centers_
        elif self.init_params == "random":
            mu = np.random.randn(self.z, n) * 0.01
        else:
            raise ValueError("init_params must be 'kmeans' or 'random'")
            
        std = np.std(x, axis=0)
        std = np.tile(std, self.z).reshape(self.z, n)
        
        mu_update = np.zeros_like(mu)
        std_update = np.ones_like(std)
        
        err_mu_res = []
        err_std_res = []
        
        for epoch in range(self.max_iter):
            # E
            for j in range(self.z):
                # 隐变量外循环
                for col in range(n):
                    # 列内循环，计算联合分布
                    pdf[:, j] *= norm.pdf(x[:, col], mu[j, col], std[j, col])
                Q[:, j] = pdf[:, j] * P[j]
            # 计算Q分布
            Q = Q / np.sum(Q, axis=1).reshape(-1, 1)
            
            # M
            for j in range(self.z):
                # 更新每个隐变量的参数值
                mu_update[j] = x.T.dot(Q[:, j]) / np.sum(Q[:, j])
                std_update[j] = np.sqrt(Q[:, j].dot((x - mu[j])**2) / np.sum(Q[:, j]))
                P[j] = np.mean(Q[:, j])
            
            # 停止条件
            err_mu = np.sum(np.abs(mu - mu_update))
            err_std = np.sum(np.abs(std - std_update))
            
            if self.verbose:
                print(f"epoch {epoch}  err_mu {err_mu}  err_std {err_std}")
            
            err_mu_res.append(err_mu)
            err_std_res.append(err_std)
            
            if err_mu <= self.tol and err_std <= self.tol:
                break
            else:
                mu = copy.deepcopy(mu_update)
                std = copy.deepcopy(std_update)
        return Q, P, err_mu_res, err_std_res
    
    
    def predict(self, Q):
        '''判断类别'''
        labels = np.argmax(Q, axis=1)
        return labels
        
                

if __name__ == "__main__":
    file_path = os.getcwd()
    dataSet = pd.read_csv(file_path + "/swiss.csv")
    x = dataSet[["Fertility", "Agriculture", "Catholic", "InfantMortality"]].values
    
    # 手写模型
    model = GMM(z=3, tol=0.001, max_iter=1000, init_params="kmeans", verbose=True)
    Q, P, err_mu_res, err_std_res = model.fit(x)
    labels = model.predict(Q)
    
    score = silhouette_score(x, labels, metric="euclidean")
    print(f"手写模型轮廓系数 {score}")
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(err_mu_res)
    plt.xlabel("epoch")
    plt.ylabel("error")
    plt.title("mu")
    plt.show()
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(err_std_res)
    plt.xlabel("epoch")
    plt.ylabel("error")
    plt.title("std")
    plt.show()
    
    # sklearn
    gmm = GaussianMixture(n_components=4, tol=0.001, max_iter=1000, init_params="kmeans")
    gmm.fit(x)
    gmm.predict_proba(x)
    labels = gmm.predict(x)
    print(f"sklearn 模型轮廓系数 {score}")
    
