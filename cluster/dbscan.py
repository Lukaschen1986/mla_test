# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import seaborn as sns

# pca
pca = PCA(n_components=2, whiten=True, svd_solver="auto") # PCA + 白化
pca.fit(variation)
data_pca = pca.transform(data)

# sklearn
dbscan = DBSCAN(eps=1, min_samples=3, metric="euclidean", n_jobs=-1)
dbscan.fit(data_pca)
labels = dbscan.labels_

##############################################################################
# 计算向量距离
def dist(x1, x2):
    res = np.sqrt(np.sum((x1 - x2)**2))
    return res

# 计算距离是否在半径内
def eps_neighbor(x1, x2, eps):
    res = dist(x1, x2) < eps
    return res

# 计算在半径内点的id
def region_query(data, pid, eps):
    n = data.shape[0]
    seeds = []
    for i in range(n):
        if eps_neighbor(data[pid,:], data[i,:], eps):
            seeds.append(i)
        else:
            continue
    return seeds

# 计算是否成功分类
def expand_cluster(data, labels, pid, cls_id, eps, min_samples):
    # 计算pid点半径内所有点的id
    seeds = region_query(data, pid, eps) 
    # 如果不满足min_samples条件，则记为噪声点
    if len(seeds) < min_samples: 
        labels[pid] = noise
        return False
    # 如果满足min_samples条件，则标记上类别号
    else:
        labels[pid] = cls_id 
        # 并将半径内的所有点都标记上该类别号
        for sid in seeds:
            labels[sid] = cls_id 
        # 基于当前pid点的邻域继续扩张
        while len(seeds) > 0:
            # 取半径内所有点的第一个点
            current_point = seeds[0] 
            # 计算新点半径内的所有点的id
            seeds_2 = region_query(data, current_point, eps) 
            # 如果满足min_samples条件，则依次循环
            if len(seeds_2) >= min_samples: 
                for i in range(len(seeds_2)):
                    pid_2 = seeds_2[i]
                    # 如果半径内的点未被标记过，则添加标记，并把索引号添加进当前pid所属的邻域
                    if labels[pid_2] == unclassified: 
                        labels[pid_2] = cls_id
                        seeds.append(pid_2)
                    # 如果半径内的点曾经被标记为噪声，则更新为新的标记，但不添加进邻域
                    elif labels[pid_2] == noise: 
                        labels[pid_2] = cls_id
                    # 如果半径内的点已被标记为某一类别，则保持不动
                    else:
                        continue
            # 如果不满足min_samples条件，则跳过
            else:
                continue
            # 剔除第一个已经计算过的点
            seeds = seeds[1:] 
        return True

# dbscan
unclassified = False; noise = 0; eps=1; min_samples=3
def dbscan(data, eps, min_samples):
    cls_id = 1
    n = data.shape[0]
    labels = [unclassified] * n # 类别标记初始化
    # 依次循环所有样本点
    for pid in range(n):
        if labels[pid] == unclassified:
            if expand_cluster(data, labels, pid, cls_id, eps, min_samples):
                cls_id += 1
            else:
                continue
        else:
            continue
    return labels
labels = dbscan(data_pca, eps, min_samples)

# plot
df_pca = pd.DataFrame(data_pca, columns=["x1","x2"])
df_pca["labels"] = labels

plt.subplots(figsize=(10, 8))
sns.scatterplot(x="x1", y="x2", hue="labels", data=df_pca)
