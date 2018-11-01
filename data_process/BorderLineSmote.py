# -*- coding: utf-8 -*-
# https://zhuanlan.zhihu.com/p/44055312 # 浅谈SMOTE之类不平衡过采样方法
# https://blog.csdn.net/a358463121/article/details/52304670 # 不平衡数据分类算法介绍与比较
# https://zhuanlan.zhihu.com/p/41237940 # 机器学习之类别不平衡问题 (3) —— 采样方法
from scipy.spatial import distance
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


class BorderLineSmote(object):
    def __init__(self, x, y, minor_label, major_label, ctn_name, bry_name, dsc_name):
        self.x = x
        self.y = y
        self.minor_label = minor_label
        self.major_label = major_label
        self.ctn_name = ctn_name
        self.bry_name = bry_name
        self.dsc_name = dsc_name
    
    
    # 计算每个少数类样本和全样本的距离
    def get_dist(self):
        # 划分连续变量、二分类变量、多分类变量
        standard_scale = lambda x: (x-x.mean()) / x.std()
        x_ctn = self.x[self.ctn_name].apply(standard_scale, axis=0) # 连续变量标准化
        x_bry = self.x[self.bry_name]
        x_dsc = pd.get_dummies(self.x[self.dsc_name], columns=self.dsc_name, drop_first=False) # 多分类变量 one-hot
        # 划分少数类样本
        x_minor = self.x[self.y == self.minor_label]
        x_minor_ctn = x_ctn[self.y == self.minor_label] # 取标准化后的数据
#        x_minor_ctn = x_minor[self.ctn_name]
        x_minor_biry = x_minor[self.bry_name]
        x_minor_dsc = x_minor[self.dsc_name] 
        x_minor_dsc = pd.get_dummies(x_minor_dsc[self.dsc_name], columns=self.dsc_name, drop_first=False) # 多分类变量 one-hot
        idx_minor = x_minor_ctn.index.tolist()
        # 划分多数类样本
        x_major = self.x[self.y == self.major_label]
        y_major = self.y[self.y == self.major_label]
#        x_major_ctn = x_major[self.ctn_name]
#        x_major_biry = x_major[self.bry_name]
#        x_major_dsc = x_major[self.dsc_name]
#        x_major_dsc = pd.get_dummies(x_major_dsc[self.dsc_name], columns=self.dsc_name, drop_first=False)
        idx_major = x_major.index.tolist()
        # 计算连续变量欧氏距离
        min_max_scale = lambda x: (x-x.min()) / (x.max()-x.min())
        dist_minor_ctn = distance.cdist(x_minor_ctn, x_ctn, metric="euclidean")
        dist_minor_ctn = pd.DataFrame(dist_minor_ctn, index=x_minor_ctn.index, columns=x_ctn.index)
        dist_minor_ctn = dist_minor_ctn.apply(min_max_scale, axis=1) # 按行归一化
        # 计算二元变量jaccard距离
        dist_minor_biry = distance.cdist(x_minor_biry, x_bry, metric="jaccard")
        dist_minor_biry = pd.DataFrame(dist_minor_biry, index=x_minor_biry.index, columns=x_bry.index)
        # 计算多元变量余弦距离
        dist_minor_dsc = distance.cdist(x_minor_dsc, x_dsc, metric="cosine")
        dist_minor_dsc = pd.DataFrame(dist_minor_dsc, index=x_minor_dsc.index, columns=x_dsc.index)
        # 合并距离数据
        dist_minor = dist_minor_ctn + dist_minor_biry + dist_minor_dsc
        return dist_minor, idx_minor, idx_major, x_minor, x_major, y_major
    
    
    # 计算每个少数类样本的k近邻，并排序
    def get_kneighbors(self, dist_minor, k):
        df_kneighbors = pd.DataFrame()
        for idx in dist_minor.index:
            kneighbors = dist_minor.loc[idx,:].sort_values(ascending=True)[1:k+1].index
            kneighbors = pd.DataFrame(kneighbors[np.newaxis,:], index=[idx])
            df_kneighbors = pd.concat((df_kneighbors, kneighbors), axis=0)
        return df_kneighbors
    
    
    # 计算每个少数类样本的近邻分别归属于少数类和多数类的个数，确定标记类型
    def get_label(self, df_kneighbors, idx_minor, idx_major, x_minor):
        label_res = np.array(())
        for idx in df_kneighbors.index:
            # 计算每个少数类样本的近邻分别归属于少数类和多数类的个数
            kneighbors = df_kneighbors.loc[idx,:] 
            n_minor = len(set(kneighbors).intersection(idx_minor))
            n_major = len(set(kneighbors).intersection(idx_major))
            # 判断标签
            if n_minor == 0:
                label = "noise"
            elif n_minor <= n_major:
                label = "danger"
            elif n_minor > n_major:
                label = "safe"
            else:
                raise ValueError("label isn't either noise, danger or safe")
            label_res = np.append(label_res, label)
        # x_minor_danger
        x_minor_danger = x_minor[label_res == "danger"]
        # epochs
        epochs = int((len(idx_major) / len(idx_minor)) - 1)
        return x_minor_danger, epochs
    
    
    # 生成新的少数类样本
    def get_minor_new(self, x_minor_danger, epochs, idx_minor, x_minor, df_kneighbors):
        x_minor_add = pd.DataFrame() # 生成新的少数类样本
        x_minor_new = pd.DataFrame() # 合并新的少数类样本
        
        for epoch in range(epochs):
            for idx in x_minor_danger.index:
                # 对每个danger样本，找出他的最近邻中属于少数类的样本，并随机抽取一个（boradline-2：对他的最近邻中任意样本随机抽取一个）
                kneighbors = df_kneighbors.loc[idx,:]
                kneighbors_intersect = sorted(set(kneighbors).intersection(idx_minor))
                kneighbors_select = np.random.choice(kneighbors_intersect, 1)[0]
                # 连续变量插值
                xi_ctn = x_minor_danger.loc[idx, self.ctn_name]
                xj_ctn = x_minor.loc[kneighbors_select, self.ctn_name]
                xij_ctn = xi_ctn + np.random.rand() * (xj_ctn - xi_ctn)
                # 二分类变量新生
                proba = x_minor[self.bry_name].apply(np.mean, axis=0).tolist() # 值为1的占比分布
                xij_bry = np.array(())
                for p in proba:
                    val = np.random.choice(a=(0,1), size=1, p=[1-p, p]) # 按照p的分布选择0或1
                    xij_bry = np.append(xij_bry, val)
                # 多分类变量新生
                xij_dsc = np.array(())
                for col in self.dsc_name:
                    val_count = x_minor[col].value_counts() / len(x_minor[col])
                    val_count = val_count.sort_index(ascending=True)
                    val = np.random.choice(a=val_count.index, size=1, p=val_count) # 按照p的分布生成多分类变量
                    xij_dsc = np.append(xij_dsc, val)
                # 合并变量
                xij = np.concatenate((xij_ctn, xij_bry, xij_dsc))
                xij = pd.DataFrame(xij[np.newaxis,:])
                x_minor_add = pd.concat((x_minor_add, xij), axis=0)
        # 合成新样本
        x_minor_add.columns = x_minor.columns
        x_minor_new = pd.concat((x_minor, x_minor_add), axis=0, ignore_index=True)
        y_minor_new = pd.Series(np.tile(1, len(x_minor_new)))
        return x_minor_new, y_minor_new
        
        
    # get_data_new
    def get_data_new(self, x_major, x_minor_new, y_major, y_minor_new):
        x_new = pd.concat((x_major, x_minor_new), axis=0, ignore_index=True)
        y_new = pd.concat((y_major, y_minor_new))
        return x_new, y_new
        
        
if __name__ == "__main__":
    x, y = make_classification(n_samples=1000, n_features=5, n_classes=2, weights=[0.9, 0.1]) # 连续
    x6 = np.random.randint(low=0, high=2, size=(1000,5)) # 二元
    x7 = np.random.randint(low=0, high=3, size=(1000,1)) # 三元
    x8 = np.random.randint(low=1, high=8, size=(1000,1)) # 多元
    x = np.concatenate((x, x6, x7, x8), axis=1)
    
    x = pd.DataFrame(x, columns=["x1","x2","x3","x4","x5","x6","x7","x8","x9","x10","x11","x12"])
    y = pd.Series(y, name="y")
    
    x.x6 = x.x6.astype(np.int64)
    x.x7 = x.x7.astype(np.int64)
    x.x8 = x.x8.astype(np.int64)
    x.x9 = x.x9.astype(np.int64)
    x.x10 = x.x10.astype(np.int64)
    x.x11 = x.x11.astype(np.int64)
    x.x12 = x.x12.astype(np.int64)

    BLS = BorderLineSmote(x, y, 
                          minor_label=1, major_label=0, 
                          ctn_name=["x1","x2","x3","x4","x5"], 
                          bry_name=["x6","x7","x8","x9","x10"], 
                          dsc_name=["x11","x12"])
    dist_minor, idx_minor, idx_major, x_minor, x_major, y_major = BLS.get_dist()
    df_kneighbors = BLS.get_kneighbors(dist_minor, k=10)
    x_minor_danger, epochs = BLS.get_label(df_kneighbors, idx_minor, idx_major, x_minor)
    x_minor_new, y_minor_new = BLS.get_minor_new(x_minor_danger, epochs, idx_minor, x_minor, df_kneighbors)
    x_new, y_new = BLS.get_data_new(x_major, x_minor_new, y_major, y_minor_new)
 
