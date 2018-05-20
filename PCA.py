# -*- coding: utf-8 -*-
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

nrow, ncol = 30, 4
M = np.random.uniform(low=-20, high=20, size=nrow*ncol).reshape(nrow,ncol)

# sklearn
scaler = StandardScaler(with_mean=True, with_std=True) # 归一化
scaler.fit(M)
M_sca = scaler.transform(M)

pca = PCA(n_components=2, whiten=True, svd_solver="auto") # PCA + 白化
pca.fit(M_sca)
M_pca = pca.transform(M_sca)

print(pca.explained_variance_) 
print(pca.explained_variance_ratio_) 
print(pca.components_[0])
print(pca.components_[1])


# manual
scaler = lambda x: (x - np.mean(x, axis=0)) / np.std(x, axis=0)

#np.cov()
def get_cov(M_sca):
    mu = np.mean(M_sca, axis=0)
    nrow, ncol = M_sca.shape
    M_cov = np.zeros((ncol, ncol))
    for i in range(ncol):
        for j in range(ncol):
            M_cov[i,j] = (M_sca[:,i]-mu[i]).dot(M_sca[:,j]-mu[j]) / nrow
    return M_cov

def pca(M_sca, M_cov, n_components=2, whiten=True):
    S, V = np.linalg.eig(M_cov) # 特征值分解
    eigen_value = S[:n_components] # 特征值
    explained_variance_ratio_ = eigen_value / np.sum(S) # 方差解释比
    eigen_vector = V[:,:n_components] # 特征向量
    # 白化
    if whiten: 
        M_pca = M_sca.dot(eigen_vector)
        eps = 10**-8
        whiten_value = np.linalg.inv(np.sqrt(np.diag(eigen_value)+eps))
        M_pca = M_pca.dot(whiten_value)
    else:
        M_pca = M_sca.dot(eigen_vector)
    
    return M_pca, eigen_value, explained_variance_ratio_, eigen_vector

M_sca = scaler(M)
M_cov = get_cov(M_sca)
#np.corrcoef(M.T)
M_pca, eigen_value, explained_variance_ratio_, eigen_vector = pca(M_sca, M_cov, n_components=2, whiten=True)
