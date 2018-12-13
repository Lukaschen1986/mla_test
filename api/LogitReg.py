# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import pickle
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


class LogitReg(object):
    def __init__(self, txt_path, filename):
        self.txt_path = txt_path
        self.filename = filename
    
    def data_load(self):
        df = pd.read_csv(self.txt_path + self.filename)
        return df
    
    def data_divide(self, df):
        df = shuffle(df, random_state=0)
        x = df.iloc[:,0:5]
        y = df.iloc[:,5]
        return x, y
    
    def data_split(self, x, y, ts):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=ts, random_state=0)
        return x_train, y_train, x_test, y_test
        
    def data_transform(self, x_train):
        std_scale = StandardScaler(with_mean=True, with_std=True)
        process_fit = std_scale.fit(x_train)
        x_train_new = process_fit.transform(x_train)
        return x_train_new, process_fit
    
    def get_transform(self, x, process_fit):
        x_new = process_fit.transform(x)
        return x_new
    
    def train(self, x_train_new, y_train):
        clf = LogisticRegression(fit_intercept=True, solver="lbfgs", n_jobs=-1, max_iter=5000, tol=10**-4)
        clf_fit = clf.fit(x_train_new, y_train)
        return clf_fit
    
    def test(self, clf_fit, x_test, y_test):
        y_hat = clf_fit.predict_proba(x_test)
        y_pred = np.argmax(y_hat, axis=1)
        y_pred[y_pred == 0] = -1
        accu = sum(y_pred == y_test)/len(y_test)
        return accu


if __name__ == "__main__":
    txt_path = "D:/my_project/Python_Project/iTravel/flask/txt/"
    LR = LogitReg(txt_path, "swiss2.csv")
    # process
    df = LR.data_load()
    x, y = LR.data_divide(df)
    x_train, y_train, x_test, y_test = LR.data_split(x, y, ts=0.2)
    x_train_new, process_fit = LR.data_transform(x_train)
    # train
    clf_fit = LR.train(x_train_new, y_train)
    # test
    x_test_new = LR.get_transform(x_test, process_fit)
    accu = LR.test(clf_fit, x_test_new, y_test)
    
    with open(txt_path + "process_fit.txt", "wb") as f:
        pickle.dump(process_fit, f)
    
    with open(txt_path + "clf_fit.txt", "wb") as f:
        pickle.dump(clf_fit, f)
    
