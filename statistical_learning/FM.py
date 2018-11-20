# -*- coding: utf-8 -*-
# https://blog.csdn.net/Chloezhao/article/details/53465167
# https://github.com/coreylynch/pyFM
# https://blog.csdn.net/anshuai_aw1/article/details/83749395
from __future__ import absolute_import, division, print_function
import os
os.getcwd()
import pickle
import copy
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer


class FM(object):
    def __init__(self, k, max_iter, lam, eta, tol):
        self.k = k
        self.max_iter = max_iter
        self.lam = lam
        self.eta = eta
#        self.decay = decay
        self.tol = tol
    
    
    def fit(self, x, y):
        # 单位向量
        unit_vector = lambda x: x / np.sqrt(np.sum(x**2, axis=0))
        # 参数初始化
        nrow, ncol = x.shape
        # w初始化
        if np.linalg.det(x.T.dot(x) + np.eye(ncol)*self.lam) == 0:
            w = np.zeros((ncol, 1)) + 0.001
        else:
            w = np.linalg.inv(x.T.dot(x) + np.eye(ncol)*self.lam).dot(x.T).dot(y)
        # b初始化
        b = np.zeros((1, 1))
        # v初始化
        v = np.random.randn(ncol, self.k) * 0.01
        # 迭代训练
        obj_log = []; gv_total = 0
        for i in range(self.max_iter):
            z = b + x.dot(w).T + np.sum(x.dot(v)**2-(x**2).dot(v**2), axis=1) / 2.0
        #    z = b + x.dot(w).T
            z = z.T
            # 计算梯度
            gb = -y.T.dot(1/(1+np.exp(y*z))) / nrow
            gw = (-y*x).T.dot(1/(1+np.exp(y*z))) / nrow + self.lam*w/nrow
            gw = unit_vector(gw)
            gv = x.T.dot(x.dot(v) - (x**2).dot(v))
            gv = unit_vector(gv)
            gv_total += gv
            # 梯度下降
            b -= self.eta * gb
            w -= self.eta * gw
            v -= self.eta / np.sqrt(1+np.sum(gv_total**2)) * gv
        #    v -= eta*(1-decay)**i * gv
            # 目标函数值
            obj = np.sum(np.log(1+np.exp(-y*z))) / nrow + self.lam*np.sum(w**2)/nrow/2.0
            obj_log.append(obj)
            # break condition
            if (i >= 1) and (np.abs(obj_log[i]-obj_log[i-1]) <= self.tol):
                break
        return obj_log, b, w, v
        
    
    def predict(self, x, b, w, v, threshold):
        z_pred = b + x.dot(w).T + np.sum(x.dot(v)**2-(x**2).dot(v**2), axis=1) / 2.0
        z_pred = z_pred.T
        y_pred = 1 / np.log(1+np.exp(-z_pred))
        y_hat = np.where(y_pred > threshold, 1, -1).T[0]
        return y_hat
    
    
    def metrics(self, y_true, y_hat):
        accu = sum(y_hat == y_true) / len(y_true)
        return accu

if __name__ == "__main__":
    #txt_path = "D:\\my_project\\Python_Project\\quant\\"
    txt_path = "D:\\my_project\\Python_Project\\iTravel\\itravel_mem_label\\script\\"
    
    df = pd.read_csv(txt_path + "mtcars2.csv")
    df.hp = df.hp.astype(np.float64)
    df = shuffle(df, random_state=0)
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=0) # test_sizes=0.3
    
    mapper = DataFrameMapper(
            features=[
                    (["mpg"], None),
                    (["disp"], None),
                    (["hp"], None),
                    (["drat"], None),
                    (["wt"], None),
                    (["qsec"], None),
                    (["vs"], LabelBinarizer()),
                    (["am"], LabelBinarizer()),
                    (["gear"], OneHotEncoder()),
                    (["carb"], OneHotEncoder()),
                    (["cyl"], OneHotEncoder()),
                    ],
            default=False # None 保留; False 丢弃
            )

    mapper_fit = mapper.fit(df_train)
    df_train_transform = mapper_fit.transform(df_train)
    df_test_transform = mapper_fit.transform(df_test)
    
    feat_name = pd.get_dummies(df_train, columns=["gear","carb","cyl"], drop_first=False, dummy_na=False).columns[1:]
    
    x_train = df_train_transform[:,1:]
    y_train = df_train_transform[:,0]
    x_test = df_test_transform[:,1:]
    y_test = df_test_transform[:,0]
    
    avg = np.mean(x_train[:,0:5], axis=0)
    std = np.std(x_train[:,0:5], axis=0)
    x_train[:,0:5] = (x_train[:,0:5] - avg) / std
    x_test[:,0:5] = (x_test[:,0:5] - avg) / std
    
    fm = FM(k=4, max_iter=2000, lam=0.0001, eta=0.01, tol=10**-8)
    obj_log, b, w, v = fm.fit(x_train, y_train)
    plt.plot(obj_log); min(obj_log)
    y_hat = fm.predict(x_test, b, w, v, threshold=0.5)
    accu = fm.metrics(y_test, y_hat)
    print(accu)
