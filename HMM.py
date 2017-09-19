# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

df = pd.read_csv("seatbelts.csv")
idx = np.random.permutation(df.index)
df = pd.DataFrame(df, index=idx)
df.index = range(len(df))

#x_name = "price"; y_name = "kill"
def data_define(df, x_name, y_name):
    # define x y
    x, y = df[x_name], df[y_name]
    # define transition matrix
    A_name = set(x); A_shape = len(A_name)
    A = pd.DataFrame(np.zeros((A_shape, A_shape)), index=A_name, columns=A_name)
    for i in range(len(df)-1):
        j = i + 1
        A.loc[x[i], x[j]] += 1
    A_sum = A.apply(np.sum, axis=1)
    for j in range(len(A)):
        A.iloc[j,:] = A.iloc[j,:] / A_sum[j]
    # define emission matrix
    B_name = set(y); B_shape = len(B_name)
    B = pd.DataFrame(np.zeros((A_shape, B_shape)), index=A_name, columns=B_name)
    for i in range(len(df)):
        B.loc[x[i], y[i]] += 1
    B_sum = B.apply(np.sum, axis=1)
    for j in range(len(B)):
        B.iloc[j,:] = B.iloc[j,:] / B_sum[j]
    return A, B
A, B = data_define(df, "price", "kill")
    
x0 = pd.DataFrame({"[0.0941,0.107)":[0.5], "[0.12,0.133]":[0.2], "[0.0811,0.0941)":[0.2], "[0.107,0.12)":[0.1]})

obs = df["kill"][1:101]

def HMM(A, B, x0, obs):
    update = x0
    n, t = len(A), len(obs)
    for step in range(t):
        # predict
        predict = update.dot(A)
        # update
        full_prob = 0.0
        for i in range(n):
            full_prob += B.loc[predict.columns[i], obs.iloc[step]]*predict.iloc[:,i]
        update = np.zeros((1,n))
        for j in range(n):
            update[0,j] = B.loc[predict.columns[j], obs.iloc[step]]*predict.iloc[:,j] / full_prob
        update = pd.DataFrame(update, columns=A.columns)
    predict = update.dot(A)
    max_label = predict.columns[np.argmax(np.array(predict))]
    max_value = np.max(np.array(predict))
    return predict, max_label, max_value
predict, max_label, max_value = HMM(A, B, x0, obs)


