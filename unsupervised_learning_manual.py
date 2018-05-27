# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import pandas as pd
import random as rd
from sklearn.cluster import KMeans
from scipy import stats
import copy

# load dataSet
dataSet = pd.read_csv("D:/my_project/Python_Project/udacity/creating_customer_segments-master/customers.csv")
x = dataSet.iloc[:,2:]

# log data
x_log = np.log(x)

# StandardScaler
scaler = lambda x: (x - np.mean(x, axis=0)) / np.std(x, axis=0)
x_sca = scaler(x_log)

# PCA
def get_cov(x_sca):
    x_sca = np.array(x_sca)
    mu = np.mean(x_sca, axis=0)
    nrow, ncol = x_sca.shape
    x_cov = np.zeros((ncol, ncol))
    for i in range(ncol):
        for j in range(ncol):
            x_cov[i,j] = (x_sca[:,i]-mu[i]).dot(x_sca[:,j]-mu[j]) / nrow
    return x_cov

def pca(x_sca, x_cov, n_components=2, whiten=True):
    x_sca = np.array(x_sca)
    S, V = np.linalg.eig(x_cov) # 特征值分解
    eigen_value = S[:n_components] # 特征值
    explained_variance_ratio_ = eigen_value / np.sum(S) # 方差解释比
    eigen_vector = V[:,:n_components] # 特征向量
    # 白化
    if whiten: 
        x_pca = x_sca.dot(eigen_vector)
        eps = 10**-8
        whiten_value = np.linalg.inv(np.sqrt(np.diag(eigen_value)+eps))
        x_pca = x_pca.dot(whiten_value)
    else:
        x_pca = x_sca.dot(eigen_vector)
    return x_pca, eigen_value, explained_variance_ratio_, eigen_vector

x_cov = get_cov(x_sca)
x_pca, eigen_value, explained_variance_ratio_, eigen_vector = pca(x_sca, x_cov, n_components=2, whiten=True)

# GMM
def runif(n, xmin, xmax):
    res = []
    for i in range(n):
        val = rd.uniform(xmin, xmax)
        res.append(val)
    res = np.array(res)
    return res

k = 3; nrow, ncol = x_pca.shape
km = KMeans(n_clusters=k, max_iter=5000, random_state=0)
km.fit(x_pca)
km_centers = km.cluster_centers_

mu_group = km_centers
sigma_group = np.std(x_pca, axis=0)
sigma_group = np.tile(sigma_group,k).reshape(k,ncol)

P = runif(n=k, xmin=0.0, xmax=1.0)
pdf = np.ones((nrow,k))
Q = np.zeros((nrow,k))

mu_new = np.zeros((k,ncol))
sigma_new = np.zeros((k,ncol))
P_new = np.tile(0.0, k)

tol = 0.001
max_iter = 5000

for step in range(max_iter):
    # E
    for j in range(k):
        for col in range(ncol):
            pdf[:,j] *= stats.norm.pdf(x_pca[:,col], mu_group[j,col], sigma_group[j,col])
    
    for row in range(len(pdf)):
        if sum(pdf[row,:]) == 0.0:
            pdf[row,:] = 0.001
            
    for i in range(nrow): 
        Q[i,:] = pdf[i,:]*P/np.dot(pdf[i,:],P)
        
    # M
    for j in range(k):
        mu_new[j,:] = np.dot(Q[:,j],x_pca)/np.sum(Q[:,j])
        sigma_new[j,:] = np.sqrt(np.dot(Q[:,j],(x_pca-mu_group[j,:])**2)/np.sum(Q[:,j]))
        P_new[j] = np.sum(Q[:,j])/nrow
        
    # epsilon
    e_mu = np.array(np.sum(np.abs(mu_new-mu_group))).sum() / ncol
    e_sigma = np.array(np.sum(np.abs(sigma_new-sigma_group))).sum() / ncol
    e_P = np.sum(np.abs(P_new-P))
    
    if e_mu <= tol and e_sigma <= tol and e_P <= tol:
        break
    else:
        mu_group = copy.deepcopy(mu_new)
        sigma_group = copy.deepcopy(sigma_new)
        P = copy.deepcopy(P_new)

gmm_labels = np.argmax(Q, axis=1)
gmm_res = np.concatenate((Q, gmm_labels[:,np.newaxis]), axis=1)
pd.DataFrame(gmm_res).to_csv("D:/my_project/Python_Project/udacity/creating_customer_segments-master/gmm_res_2.csv")

fig=10; ax=6
cluster_result(x_pca, gmm_labels, mu_group, fig, ax)

def silhouette_samples_manual(X, labels):
    df = pd.DataFrame(np.concatenate((labels[:,np.newaxis], X), axis=1))
    df = df.set_index(0)
    
    dist_1 = np.sum((df.iloc[0,:]-df)**2, axis=1)
    dist_1.groupby(dist_1.index).mean()
