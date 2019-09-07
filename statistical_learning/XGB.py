# -*- coding: utf-8 -*-
import os
import random as rd
import numpy as np
import pandas as pd
#from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
#import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost.sklearn import (XGBRegressor, XGBClassifier)


class XGBoostModel(object):
    def __init__(self, job_type, learning_rate, alpha, lamb, min_sample_split,
                 subsample, colsample, n_estimators, use_early_stopping, tol,
                 verbose):
        self.job_type = job_type
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.lamb = lamb
        self.min_sample_split = min_sample_split
        self.subsample = subsample
        self.colsample = colsample
        self.n_estimators = n_estimators
        self.use_early_stopping = use_early_stopping
        self.tol = tol
        self.verbose = verbose
    
    
    def check_params(self):
        '''参数校验'''
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must > 0, float")
        if self.alpha < 0:
            raise ValueError("alpha must >= 0, default = 0")
        if self.lamb < 0:
            raise ValueError("lambda must >= 0, default = 1")
        if self.min_sample_split <= 0:
            raise ValueError("min_sample_split must > 1, int")
        if self.subsample <= 0 or self.subsample > 1:
            raise ValueError("subsample range (0, 1], float")
        if self.colsample <= 0 or self.colsample > 1:
            raise ValueError("colsample range (0, 1], float")
        if self.n_estimators < 1:
            raise ValueError("n_estimators must >= 1, int")
        return None
    
    
    def data_augment(self, x):
        '''
        数据增强：
        1、列名改为数字
        2、为每个连续变量加上一个极端小量 N(0, 10**-8)，使每个连续变量值唯一，
        从而可以避免后面建树时出现变量值相同而分裂点报错
        '''
        N, _ = x.shape
        x.columns = range(len(x.columns))
        
        for j in x.columns:
            if isinstance(x[j].iloc[0], np.float64):
                randn = pd.Series(np.random.normal(0.0, 10**-8, N), index=x.index)
                x[j] += randn
            else:
                continue
        return x
    
    
    def data_sampling(self, x, y):
        '''
        样本抽样：
        1、行抽样
        2、列抽样
        3、根据抽样提取子集
        '''
        # 行抽样 & 列抽样
        idx_row = rd.sample(x.index.tolist(), int(len(x) * self.subsample))
        idx_col = rd.sample(x.columns.tolist(), int(len(x.columns) * self.colsample))
        # 提取子集
        x_sub = x.loc[idx_row, idx_col]
        y_sub = y[idx_row]
        return x_sub, y_sub
    
    
    def get_info(self, g, h):
        '''计算信息增益'''
        fenzi = (g.sum() + self.alpha)**2
        fenmu = h.sum() + self.lamb
        info = fenzi / (fenmu + 10**-8)
        return info
    
    def get_w(self, g, h):
        '''计算样本落到叶子结点的得分，即假设函数'''
        fenzi = g.sum() + self.alpha
        fenmu = h.sum() + self.lamb
        weight = -fenzi / (fenmu + 10**-8)
        return weight
    
    def split_data(self, x, g, h, best_feature, best_value):
        '''根据信息增益最大化，分裂子树'''
        # 样本分裂
        x_left = x[x[best_feature] < best_value]
        x_right = x[x[best_feature] >= best_value]
        # 一阶导分裂
        g_left = g[x[best_feature] < best_value]
        g_right = g[x[best_feature] >= best_value]
        # 二阶导分裂
        h_left = h[x[best_feature] < best_value]
        h_right = h[x[best_feature] >= best_value]
        return x_left, x_right, g_left, g_right, h_left, h_right
    
    
    def create_tree(self, x, g, h):
        '''递归建树'''
        N, _ = x.shape
        # 判断结点最小分裂样本，如果达到阈值则不再分裂，而是将该结点置为叶子结点计算得分
        if N <= self.min_sample_split:
            wj = self.get_w(g, h)
            return wj
        # 信息增益评价表
        df_info_gain_res = pd.DataFrame()
        # 列循环
        for j in x.columns:
            # 列排序
            x = x.sort_values(by=j, ascending=True)
            g = g.reindex(index=x.index)
            h = h.reindex(index=x.index)
            # 如果特征为连续属性，则逐行搜索最优分裂点
            if isinstance(x[j].iloc[0], np.float64):
                for i in range(1, N):
                    g_left = g.iloc[0:i]
                    g_right = g.iloc[i:]
                    
                    h_left = h.iloc[0:i]
                    h_right = h.iloc[i:]
                    
                    info_parent = self.get_info(g, h)
                    info_left = self.get_info(g_left, h_left)
                    info_right = self.get_info(g_right, h_right)
                    info_gain = info_left + info_right - info_parent
                    
                    df_info_gain = pd.DataFrame({"feature": [j],
                                                 "value": x[j].iloc[i],
                                                 "info_gain": [info_gain]},
                                                 columns=["feature", "value", "info_gain"])
                    df_info_gain_res = pd.concat([df_info_gain_res, df_info_gain], 
                                                 axis=0, 
                                                 ignore_index=True)
            # 如果特征为离散属性，则当属性值发生变化时，触发搜索最优分裂点
            elif isinstance(x[j].iloc[0], np.int64):
                for i in range(1, N):
                    if x[j].iloc[i] != x[j].iloc[i-1]:
                        g_left = g.iloc[0:i]
                        g_right = g.iloc[i:]
                        
                        h_left = h.iloc[0:i]
                        h_right = h.iloc[i:]
                        
                        info_parent = self.get_info(g, h)
                        info_left = self.get_info(g_left, h_left)
                        info_right = self.get_info(g_right, h_right)
                        info_gain = info_left + info_right - info_parent
                        
                        df_info_gain = pd.DataFrame({"feature": [j],
                                                     "value": x[j].iloc[i],
                                                     "info_gain": [info_gain]},
                                                     columns=["feature", "value", "info_gain"])
                        df_info_gain_res = pd.concat([df_info_gain_res, df_info_gain], 
                                                     axis=0, 
                                                     ignore_index=True)
            else:
                raise ValueError("dtypes of data should be np.float64 or np.int64")
        # 计算增益最大化、最优特征、最优分裂值
        best_gain = df_info_gain_res.info_gain.max()
        best_feature = df_info_gain_res.loc[df_info_gain_res.info_gain == best_gain, "feature"].values[0]
        best_value = df_info_gain_res.loc[df_info_gain_res.info_gain == best_gain, "value"].values[0]
        # 构建左右子树有向边条件
        left_rule = "< " + str(best_value)
        right_rule = ">= " + str(best_value)
        # 分裂
        x_left, x_right, g_left, g_right, h_left, h_right = self.split_data(x, g, h, 
                                                                            best_feature, 
                                                                            best_value)
        # 递归建树
        tree = {best_feature: {}}
        
        try:
            tree[best_feature][left_rule] = self.create_tree(x_left, g_left, h_left)
        except Exception as e_left:
            print(e_left)
        
        try:
            tree[best_feature][right_rule] = self.create_tree(x_right, g_right, h_right)
        except Exception as e_right:
            print(e_right)
        
        return tree
        
    
    def predict_single(self, tree, x_sub):
        '''单样本预测'''
        # 递归解析建好的树
        for (feature, branch) in tree.items():
            # 读取特征值
            value = x_sub[feature]
            # 获取判断条件
            left_rule = list(branch.keys())[0]
            right_rule = list(branch.keys())[1]
            # 是否满足左分支
            if eval(str(value) + left_rule):
                tree = list(branch.values())[0]
                if isinstance(tree, np.float64):
                    return tree
                else:
                    tree = self.predict_single(tree, x_sub)
            # 是否满足右分支
            if eval(str(value) + right_rule):
                tree = list(branch.values())[1]
                if isinstance(tree, np.float64):
                    return tree
                else:
                    tree = self.predict_single(tree, x_sub)
        return tree
    
    def predict_batch(self, tree, x):
        '''批预测'''
        y_hat_list = []
        for i in range(len(x)):
            x_sub = x.iloc[i, :]
            y_hat_sub = self.predict_single(tree, x_sub)
            y_hat_list.append(y_hat_sub)
        
        y_hat = pd.Series(y_hat_list, index=x.index)
        return y_hat
    
    def predict(self, forest, x):
        '''模型预测（回归）'''
        y_pred = pd.Series(np.zeros(len(x)), index=x.index)
        for i in range(len(forest)):
            y_hat = self.predict_batch(forest[i], x)
            y_pred += self.learning_rate * y_hat
        return y_pred
    
    def predict_proba(self, forest, x):
        '''模型预测（二分类）'''
        y_pred = pd.Series(np.zeros(len(x)), index=x.index) + 0.5
        for i in range(len(forest)):
            y_hat = self.predict_batch(forest[i], x)
            y_pred += self.learning_rate * y_hat
        return y_pred
    
    
    def loss_reg(self, y_true, y_pred):
        '''均方根误差'''
        return np.sqrt(np.mean((y_true - y_pred)**2))
    
    def loss_bina(self, y_true, y_pred):
        '''交叉熵'''
        res = - y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
        return np.mean(res)
    
    
    def fit(self, x_train, y_train, x_test, y_test):
        '''模型训练'''
        # 参数校验
        self.check_params()
        # 变量初始化
        if self.job_type == "Regressor":
            y_pred_train = pd.Series(np.zeros_like(y_train), index=y_train.index)
            y_pred_test = pd.Series(np.zeros_like(y_test), index=y_test.index)
            g = y_pred_train - y_train
            h = pd.Series(np.ones_like(y_train), index=y_train.index)
        elif self.job_type == "Classifier":
            y_pred_train = pd.Series(np.zeros_like(y_train), index=y_train.index) + 0.5
            y_pred_test = pd.Series(np.zeros_like(y_test), index=y_test.index) + 0.5
            g = - (y_train / y_pred_train) + ((1 - y_train) / (1 - y_pred_train))
            h = (y_train / y_pred_train**2) + ((1 - y_train) / (1 - y_pred_train)**2)
        else:
            raise ValueError("job_type must be 'Regressor' or 'Classifier'")
        
        loss_train_list = []
        loss_test_list = []
        forest = []
        
        # 迭代训练子树模型
        for epoch in range(self.n_estimators):
            # 行列抽样
            x_sub, y_sub = self.data_sampling(x_train, y_train)
            g_sub = g[x_sub.index]
            h_sub = h[x_sub.index]
            # 递归建树
            tree = self.create_tree(x_sub, g_sub, h_sub)
            # 计算训练集和测试集损失并保存
            y_hat_train = self.predict_batch(tree, x_train)
            y_hat_test = self.predict_batch(tree, x_test)
            y_pred_train += self.learning_rate * y_hat_train
            y_pred_test += self.learning_rate * y_hat_test
            # 更新变量
            if self.job_type == "Regressor":
                # 更新梯度
                g = y_pred_train - y_train
                loss_train = self.loss_reg(y_train, y_pred_train)
                loss_test = self.loss_reg(y_test, y_pred_test)
            elif self.job_type == "Classifier":
                # 预测值调整
                y_pred_train = y_pred_train.apply(lambda y: 0.01 if y < 0 else y).\
                apply(lambda y: 0.99 if y > 1 else y)
                y_pred_test = y_pred_test.apply(lambda y: 0.01 if y < 0 else y).\
                apply(lambda y: 0.99 if y > 1 else y)
                # 更新梯度
                g = - (y_train / y_pred_train) + ((1 - y_train) / (1 - y_pred_train))
                h = (y_train / y_pred_train**2) + ((1 - y_train) / (1 - y_pred_train)**2)
                loss_train = self.loss_bina(y_train, y_pred_train)
                loss_test = self.loss_bina(y_test, y_pred_test)
            else:
                raise ValueError("job_type must be 'Regressor' or 'Classifier'")
                
            # 打印结果
            if self.verbose:
                print(f"epoch {epoch}  loss_train {loss_train}  loss_test {loss_test}")
            
            loss_train_list.append(loss_train)
            loss_test_list.append(loss_test)
            forest.append(tree)
            
            # 判断停止条件并保存
            if self.use_early_stopping:
                err = loss_test_list[epoch-1] - loss_test_list[epoch]
                if len(loss_test_list) >= 2 and err <= self.tol:
                    print(f"best iteration at epoch {epoch-1}")
                    evals_result = {"loss_train": loss_train_list[:-1], 
                                    "loss_test": loss_test_list[:-1]}
                    forest = forest[:-1]
                    break
                else:
                    continue
            else:
                evals_result = {"loss_train": loss_train_list, "loss_test": loss_test_list}
                
        return forest, evals_result
            
        
    def get_score(self, y_true, y_pred):
        '''计算准确率'''
        score = sum(y_true == y_pred) / len(y_true)
        return score
        
        
        

if __name__ == "__main__":
    file_path = os.getcwd()
    dataSet = pd.read_csv(file_path + "/swiss.csv")
    
    # 1、回归问题
    # 1.1、手写算法
    df = dataSet[["Fertility", "Agriculture", "Catholic", "InfantMortality", "Examination", "Education"]]
    y = df.iloc[:, 0]
    x = df.iloc[:, 1:]
    
    model = XGBoostModel(job_type="Regressor", learning_rate=0.3, alpha=0, lamb=1, 
                         min_sample_split=1, subsample=0.8, colsample=0.8, 
                         n_estimators=100, use_early_stopping=True, tol=0.0001,
                         verbose=True)
    x = model.data_augment(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    forest, evals_result = model.fit(x_train, y_train, x_test, y_test)
    df_evals = pd.DataFrame({"loss_train": evals_result.get("loss_train"),
                             "loss_test": evals_result.get("loss_test")})
    df_evals.plot()
    y_pred = model.predict(forest, x_test)
    model.loss_reg(y_test, y_pred)
    
    # 1.2、sklearn
    d_train = xgb.DMatrix(x_train, y_train)
    d_test = xgb.DMatrix(x_test, y_test)
    wathchlist = [(d_train, "train"), (d_test, "test")]
    reg = XGBRegressor(learning_rate=0.3, n_estimators=100,
                       objective="reg:linear", eval_metric="rmse", 
                       min_child_weight=1, subsample=0.8, colsample_bytree=0.8, 
                       reg_alpha=0, reg_lambda=1, n_jobs=-1, nthread=-1, seed=3)
    params = reg.get_params()
    evals_res = {}
    model_sklearn = xgb.train(params=params, dtrain=d_train, evals=wathchlist, 
                              evals_result=evals_res, early_stopping_rounds=10, 
                              verbose_eval=True)
    y_pred = model_sklearn.predict(d_test)
    df_evals = pd.DataFrame({"loss_train": evals_res.get("train").get("rmse"),
                             "loss_test": evals_res.get("test").get("rmse")})
    df_evals.plot()
    model.loss_reg(y_test, y_pred)
    
    # 2、二分类问题
    # 2.1、手写算法
    df = dataSet[["Fertility2", "Agriculture", "Catholic", "InfantMortality", "Examination", "Education"]]
    y = df.iloc[:, 0]
    x = df.iloc[:, 1:]
    
    model = XGBoostModel(job_type="Classifier", learning_rate=0.1, alpha=0, lamb=1, 
                         min_sample_split=1, subsample=0.8, colsample=0.8, 
                         n_estimators=100, use_early_stopping=True, tol=0.0001,
                         verbose=True)
    x = model.data_augment(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    forest, evals_result = model.fit(x_train, y_train, x_test, y_test)
    df_evals = pd.DataFrame({"loss_train": evals_result.get("loss_train"),
                             "loss_test": evals_result.get("loss_test")})
    df_evals.plot()
            
    y_hat = model.predict_proba(forest, x_test)
    y_pred = y_hat.apply(lambda y: 0 if y <= 0.5 else 1)
    model.get_score(y_test, y_pred)
        
    # 2.2、sklearn
    d_train = xgb.DMatrix(x_train, y_train)
    d_test = xgb.DMatrix(x_test, y_test)
    wathchlist = [(d_train, "train"), (d_test, "test")]
    clf = XGBClassifier(learning_rate=0.1, n_estimators=100,
                       objective="binary:logistic", eval_metric="logloss", 
                       min_child_weight=1, subsample=0.8, colsample_bytree=0.8, 
                       reg_alpha=0, reg_lambda=1, n_jobs=-1, nthread=-1, seed=3)
    params = clf.get_params()
    evals_res = {}
    model_sklearn = xgb.train(params=params, dtrain=d_train, evals=wathchlist, 
                              evals_result=evals_res, early_stopping_rounds=10, 
                              verbose_eval=True)
    y_hat = model_sklearn.predict(d_test)
    df_evals = pd.DataFrame({"loss_train": evals_res.get("train").get("logloss"),
                             "loss_test": evals_res.get("test").get("logloss")})
    df_evals.plot()
    
    y_pred = np.where(y_hat <= 0.5, 0, 1)
    model.get_score(y_test, y_pred)
               
