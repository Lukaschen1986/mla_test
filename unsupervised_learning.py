# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
import matplotlib.pyplot as plt

def pca_result(x, x_pca, pca, fig, ax, arrow_size, text_pos, fontsize):
    x_pca = pd.DataFrame(x_pca, columns = ['Dimension 1', 'Dimension 2'])
    fig, ax = plt.subplots(figsize=(fig, ax))
    # scatterplot of the reduced data    
    ax.scatter(x=x_pca.loc[:, 'Dimension 1'], y=x_pca.loc[:, 'Dimension 2'], 
               facecolors='b', edgecolors='b', s=70, alpha=0.5)
    feature_vectors = pca.components_.T
    
    # projections of the original features
    for i, v in enumerate(feature_vectors):
        ax.arrow(0, 0, arrow_size*v[0], arrow_size*v[1], 
                  head_width=0.2, head_length=0.2, linewidth=2, color='red')
        ax.text(v[0]*text_pos, v[1]*text_pos, x.columns[i], color='black', 
                 ha='center', va='center', fontsize=fontsize)

    ax.set_xlabel("Dimension 1", fontsize=14)
    ax.set_ylabel("Dimension 2", fontsize=14)
    ax.set_title("PC plane with original feature projections.", fontsize=16);
    return ax

def cluster_result(x_pca, labels, centers, fig, ax):
    predictions = pd.DataFrame(labels, columns = ['Cluster'])
    x_pca = pd.DataFrame(x_pca, columns = ['Dimension 1', 'Dimension 2'])
    plot_data = pd.concat([predictions, x_pca], axis = 1)
    
    # Generate the cluster plot
    fig, ax = plt.subplots(figsize=(fig, ax))
    
    # Color map
    cmap = plt.cm.get_cmap('gist_rainbow')
    
    # Color the points based on assigned cluster
    for i, cluster in plot_data.groupby('Cluster'):
        cluster.plot(ax=ax, kind='scatter', x='Dimension 1', y='Dimension 2', 
                     color=cmap((i)*1.0/(len(centers)-1)), label='Cluster %i'%(i), s=30)
    
    # Plot centers with indicators
    for i, c in enumerate(centers):
        ax.scatter(x = c[0], y = c[1], color = 'white', edgecolors = 'black', alpha = 1, linewidth = 2, marker = 'o', s=200)
        ax.scatter(x = c[0], y = c[1], marker='$%d$'%(i), alpha = 1, s=100)
    
    # Set plot title
    ax.set_title("Cluster Learning on PCA-Reduced Data - Centroids Marked by Number\nTransformed Sample Data Marked by Black Cross")
    
    
# load dataSet
dataSet = pd.read_csv("D:/my_project/Python_Project/udacity/creating_customer_segments-master/customers.csv")
x = dataSet.iloc[:,2:]
pd.plotting.scatter_matrix(x, alpha=0.3, figsize=(10,6), diagonal = 'kde')

# log data
x_log = np.log(x)
pd.plotting.scatter_matrix(x_log, alpha=0.3, figsize=(10,6), diagonal = 'kde')

# StandardScaler
scaler = StandardScaler(with_mean=True, with_std=True)
scaler.fit(x_log)
x_sca = scaler.transform(x_log)
pd.plotting.scatter_matrix(x_sca, alpha=0.3, figsize=(10,6), diagonal = 'kde')

# PCA
pca = PCA(n_components=2, whiten=True, svd_solver="auto") # PCA + 白化
pca.fit(x_sca)
x_pca = pca.transform(x_sca)

fig=10; ax=6; arrow_size=4.0; text_pos=5.0; fontsize=10
pca_result(x, x_pca, pca, fig, ax, arrow_size, text_pos, fontsize)

k = 3
# Kmeans
km = KMeans(n_clusters=k, max_iter=5000, random_state=0)
km.fit(x_pca)
km_centers = km.cluster_centers_
km_labels = km.predict(x_pca)
print(silhouette_samples(X=x_pca, labels=km_labels))
print(silhouette_score(X=x_pca, labels=km_labels))
cluster_result(x_pca, km_labels, km_centers, fig, ax)

# GMM
gmm = GaussianMixture(n_components=k, init_params="kmeans", tol=0.001, max_iter=5000, random_state=1)
gmm.fit(x_pca)
gmm_centers = gmm.means_
gmm_labels = gmm.predict(x_pca)
gmm_proba = gmm.predict_proba(x_pca)
print(gmm.n_iter_)
print(silhouette_samples(X=x_pca, labels=gmm_labels))
print(silhouette_score(X=x_pca, labels=gmm_labels))
cluster_result(x_pca, gmm_labels, gmm_centers, fig, ax)

gmm_res = np.concatenate((gmm_proba, gmm_labels[:,np.newaxis]), axis=1)
pd.DataFrame(gmm_res).to_csv("D:/my_project/Python_Project/udacity/creating_customer_segments-master/gmm_res_1.csv")
