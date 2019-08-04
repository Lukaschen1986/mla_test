# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)
import numpy as np
import pandas as pd
import scipy.spatial.distance as dist
from scipy.stats import norm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
#import random as rd
#import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris


class NaiveBayesModel(object):
    def __init__(self, x_train, y_train, x_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.N, self.n = x_train.shape
    
    
    def prior_probs(self):
        '''计算标记值的先验概率(含拉普拉斯平滑)： 本案设为三分类问题'''
        labels, labels_count = np.unique(self.y_train, return_counts=True)
        label_0, label_1, label_2 = labels[0], labels[1], labels[2]
        prior_probs_0, prior_probs_1, prior_probs_2 = (labels_count + 1) / (labels_count.sum() + self.n)
        # 类别标记
        label_dict = {"label_0": label_0, 
                      "label_1": label_1, 
                      "label_2": label_2}
        # 类别标记对应的先验概率
        prior_probs_dict = {"prior_probs_0": prior_probs_0, 
                            "prior_probs_1": prior_probs_1, 
                            "prior_probs_2": prior_probs_2}
        return label_dict, prior_probs_dict
        

    def likelihood_probs(self, label):
        '''计算每个样本的似然概率(含拉普拉斯平滑)'''
        # 初始化似然概率数组
        likeli_probs_array = np.array([])
        # 行循环
        for i in range(len(self.x_test)):
            # 初始化似然概率的联合分布
            likeli_probs_res = 1.0
            # 列循环
            for j in range(self.n):
                val = self.x_test.iloc[i, j]
                # 若字段取值为分类变量，则计算条件概率，加拉普拉斯平滑
                if isinstance(val, str):
                    fen_zi = len(self.x_train.loc[(self.x_train.iloc[:, j] == val) & 
                                                  (y_train == label)]) + 1
                    fen_mu = sum(self.y_train == label) + self.n
                    likeli_probs = fen_zi / fen_mu
                # 若字段取值为连续变量，则依据正态分布计算条件概率密度
                elif isinstance(val, float):
                    vals = x_train[self.y_train == label].iloc[:, j]
                    mean = np.mean(vals)
                    std = np.std(vals)
                    likeli_probs = norm.pdf(val, mean, std)
                else:
                    raise TypeError("Type must be as str or float.")
                # 计算似然概率的联合分布
                likeli_probs_res *= likeli_probs
            # 保持似然概率
            likeli_probs_array = np.append(likeli_probs_array, likeli_probs_res)
        return likeli_probs_array
    
    
    def predict(self, prior_probs_0, prior_probs_1, prior_probs_2, 
                likeli_probs_array_0, likeli_probs_array_1, likeli_probs_array_2):
        '''模型预测, 计算后验概率, 取最大化'''
        posterior_probs_0 = (prior_probs_0 * likeli_probs_array_0).reshape(-1, 1)
        posterior_probs_1 = (prior_probs_1 * likeli_probs_array_1).reshape(-1, 1)
        posterior_probs_2 = (prior_probs_2 * likeli_probs_array_2).reshape(-1, 1)
        
        posterior_probs = np.concatenate([posterior_probs_0, 
                                          posterior_probs_1, 
                                          posterior_probs_2], axis=1)
        # 最大后验概率
        y_pred = np.argmax(posterior_probs, axis=1)
        return y_pred
    
    
    def get_score(self, y_true, y_pred):
        '''模型评估'''
        score = sum(y_true == y_pred) / len(y_true)
        return score
        
    
    
if __name__ == "__main__":
    # 读取数据
    iris = load_iris()
    dataSet = pd.DataFrame(iris.data, columns=iris.feature_names)
    dataSet.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    # 构造三分类变量
    dataSet["sepal_length_mult"] = pd.cut(dataSet.sepal_length, bins=3, 
           right=True, include_lowest=True, 
           labels=["low", "med", "high"])
    
    dataSet["sepal_width_mult"] = pd.cut(dataSet.sepal_width, bins=3, 
           right=True, include_lowest=True, 
           labels=["low", "med", "high"])
    # 构造二分类变量
    dataSet["petal_length_bina"] = pd.cut(dataSet.petal_length, bins=2, 
           right=True, include_lowest=True, 
           labels=["small", "big"])
    
    dataSet["petal_width_mult"] = pd.cut(dataSet.petal_width, bins=2, 
           right=True, include_lowest=True, 
           labels=["small", "big"])
    # 标记值
    dataSet["label"] = iris.target
    # 洗牌
    dataSet = shuffle(dataSet) # , random_state=0
    
    x = dataSet.iloc[:, 0:8]
    y = dataSet.iloc[:, 8]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    
    # 手写模型
    model = NaiveBayesModel(x_train, y_train, x_test)
    label_dict, prior_probs_dict = model.prior_probs()
    
    label_0 = label_dict.get("label_0")
    label_1 = label_dict.get("label_1")
    label_2 = label_dict.get("label_2")

    prior_probs_0 = prior_probs_dict.get("prior_probs_0")
    prior_probs_1 = prior_probs_dict.get("prior_probs_1")
    prior_probs_2 = prior_probs_dict.get("prior_probs_2")  
    
    likeli_probs_array_0 = model.likelihood_probs(label=label_0)
    likeli_probs_array_1 = model.likelihood_probs(label=label_1)
    likeli_probs_array_2 = model.likelihood_probs(label=label_2)
    
    y_pred = model.predict(prior_probs_0, prior_probs_1, prior_probs_2, 
                           likeli_probs_array_0, likeli_probs_array_1, likeli_probs_array_2)
    
    model.get_score(y_test, y_pred)
