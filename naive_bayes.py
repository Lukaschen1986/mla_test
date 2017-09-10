# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cross_validation import train_test_split

age = [18,22,30,40,45,50,35,21,19,55,20,32,31,60]
income = ["high","high","high","medium","low","low","low","medium","low","medium","medium","medium","high","medium"]
student = ["no","no","no","no","yes","yes","yes","no","yes","yes","yes","no","yes","no"]
credit_rating = ["fair","excellent","fair","fair","fair","excellent","excellent","fair","fair","fair","excellent","excellent","fair","excellent"]
buys_computer = ["0","0","1","1","1","0","1","0","1","1","1","1","1","0"]
df = pd.DataFrame({"age": age, 
                   "income": income, 
                   "student":student, 
                   "credit_rating":credit_rating, 
                   "buys_computer":buys_computer}, columns=["age","income","student","credit_rating","buys_computer"])
df_train, df_predict = train_test_split(df, test_size=0.1)
df_train.index = range(len(df_train))
df_predict.index = range(len(df_predict))

# y = "no"
def prior_prob(df_train, y_train, y_label):
    count = df_train[y_train].value_counts()[y_label]
    res = count/len(df_train)
    return res


def condi_prob(df_train, y_train, y_label, df_predict):
    df_sub = df_train[df_train[y_train] == y_label]
    n = len(df_predict); m = len(df_train.columns)-1
    prob = np.zeros((n,1))
    for i in range(n):
        for j in range(m):
            if isinstance(df_predict.iloc[i,j], str):
                prob[i] += np.log((np.sum(df_predict.iloc[i,j] == df_sub.iloc[:,j])+1)/(len(df_sub)+m))
            else:
                prob[i] += np.log(stats.norm.pdf(df_predict.iloc[i,j],np.mean(df_sub.iloc[:,j]),np.std(df_sub.iloc[:,j])))
    probs = np.exp(prob)
    return probs
   

#def margin_prob(df_train, df_predict):
#    n = len(df_predict); m = len(df_train.columns)-1
#    prob = np.zeros((n,1))
#    for i in range(n):
#        for j in range(m):
#            if isinstance(df_predict.iloc[i,j], str):
#                prob[i] += np.log((np.sum(df_predict.iloc[i,j] == df_train.iloc[:,j])+1)/(len(df_train)+m))
#            else:
#                prob[i] += np.log(stats.norm.pdf(df_predict.iloc[i,j],np.mean(df_train.iloc[:,j]),np.std(df_train.iloc[:,j])))
#    probs = np.exp(prob)
#    return probs

def predict_func(df_train, y_train, y_label_set, df_predict):
    n = len(df_predict)
    pred = np.zeros((n,1))
    for y_label in y_label_set: 
        # y_label_set = ["0","1"]
        joint_prob = condi_prob(df_train, y_train, y_label, df_predict)*prior_prob(df_train, y_train, y_label)
        pred = np.concatenate((pred, joint_prob), axis=1)
    pred = pred[:,1:]
    pred_sca = pred/np.sum(pred, axis=1, keepdims=True)
    y_hat = np.argmax(pred_sca, axis=1)
    return y_hat
    

