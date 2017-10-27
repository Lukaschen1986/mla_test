# -*- coding: utf-8 -*-
from __future__ import division
import os
os.getcwd()
import numpy as np
import pandas as pd
import random as rd
from sklearn import datasets
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

digits = datasets.load_digits()
data = digits.data
target = digits.target

#def data_split(data, prob):
#    idx_train = rd.sample(range(len(data)), int(len(data)*prob))
#    idx_test = [i for i in range(len(data)) if i not in idx_train]
#    return idx_train, idx_test

# scale
eps = 10**-8
X_0, X_predict_0, y, y_predict = train_test_split(data, target, test_size=0.4)
X = (X_0-np.mean(X_0, axis=0, keepdims=True)) / (np.std(X_0, axis=0, keepdims=True)+eps)
X_predict = (X_predict_0-np.mean(X_0, axis=0, keepdims=True)) / (np.std(X_0, axis=0, keepdims=True)+eps)
#X, X_predict, y, y_predict = train_test_split(data, target, test_size=0.4)

# active_func
def sigmiod(z):
    res = 1.0 / (1.0 + np.exp(-z))
    return res
    
def tanh(z):
    res = 2.0 / (1.0 + np.exp(-z)) - 1
    return res
    
def relu(z):
    res = np.maximum(0, z)
    return res
    
def leaky_relu(z):
    res = np.maximum(0.01*z, z)
    return res
    
def softmax(z):
    res = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
    return res
    
def tanh_dv(z):
    res = 1-(np.tanh(z))**2
    return res
    
def relu_dv(z):
    res = np.where(z < 0, 0, 1)
    return res
    
def leaky_relu_dv(z):
    res = np.where(z < 0, 0.01, 1)
    return res

def unit_vec(x):
    res = x/np.sqrt(np.sum(x**2))
    return res

def gradient_descent(alpha, param, param_dv):
    param -= alpha*unit_vec(param_dv)
    return param

def Momentum(w, b, dw, db, v_dw, v_db, beta_1, alpha):
    # v
    v_dw = beta_1*v_dw + (1-beta_1)*dw
    v_db = beta_1*v_db + (1-beta_1)*db
    # gradient_decent
    w -= alpha * v_dw
    b -= alpha * v_db
    return w, b, v_dw, v_db

def Adam(w, b, dw, db, v_dw, v_db, s_dw, s_db, beta_1, beta_2, alpha, eps, i, is_corr):
    # v & s
    v_dw = beta_1*v_dw + (1-beta_1)*dw
    v_db = beta_1*v_db + (1-beta_1)*db
    s_dw = beta_2*s_dw + (1-beta_2)*(dw**2)
    s_db = beta_2*s_db + (1-beta_2)*(db**2)
    # corr
    if is_corr == "yes":
        v_dw_corr = v_dw/(1-beta_1**i)
        v_db_corr = v_db/(1-beta_1**i)
        s_dw_corr = s_dw/(1-beta_2**i)
        s_db_corr = s_db/(1-beta_2**i)
    elif is_corr == "no":
        v_dw_corr = v_dw
        v_db_corr = v_db
        s_dw_corr = s_dw
        s_db_corr = s_db
    else:
        raise ValueError("is_corr must in ('yes','no')")
    # gradient_decent
    w -= alpha * v_dw_corr/(np.sqrt(s_dw_corr)+eps)
    b -= alpha * v_db_corr/(np.sqrt(s_db_corr)+eps)
    return w, b, v_dw_corr, v_db_corr, s_dw_corr, s_db_corr

# alpha_decay
def alpha_decay_1(alpha, epoch, decay_rate):
    alpha_update = decay_rate**epoch * alpha
    return alpha_update

def alpha_decay_2(alpha, epoch, decay_rate):
    alpha_update = 1.0 / (1.0 + decay_rate*epoch) * alpha
    return alpha_update

val_1 = []; val_2 = []; iters = 200
for i in range(iters):
    val = alpha_decay_1(0.1, i, 0.98)
    val_1.append(val)
for j in range(iters):
    val = alpha_decay_2(0.1, j, 0.01)
    val_2.append(val)
alpha_decay = pd.DataFrame({"val_1": val_1, "val_2": val_2})
alpha_decay.plot(xlim=[-2,iters])

# drop_out
keep_prob = 0.8
def drop_out(a, keep_prob):
    d = np.random.rand(a.shape[0], a.shape[1]) < keep_prob
    a *= d
    a /= keep_prob
    return a
#np.longfloat( 1.0/(1+np.exp(np.inf)))

x = len(X)
p = X.shape[1]

hideNum_1 = 70
hideNum_2 = 70
hideNum_3 = 70
outNum = len(set(y))

#w1 = np.random.randn(p, hideNum_1) * 0.01
#w2 = np.random.randn(hideNum_1, hideNum_2) * 0.01
#w3 = np.random.randn(hideNum_2, hideNum_3) * 0.01
#w4 = np.random.randn(hideNum_3, outNum) * 0.01

w1 = np.random.randn(p, hideNum_1) * np.sqrt(2.0/p)
w2 = np.random.randn(hideNum_1, hideNum_2) * np.sqrt(2.0/hideNum_1)
w3 = np.random.randn(hideNum_2, hideNum_3) * np.sqrt(2.0/hideNum_2)
w4 = np.random.randn(hideNum_3, outNum) * np.sqrt(2.0/hideNum_3)
b1 = np.zeros((1, hideNum_1))
b2 = np.zeros((1, hideNum_2))
b3 = np.zeros((1, hideNum_3))
b4 = np.zeros((1, outNum))

v_dw1 = s_dw1 = np.zeros((w1.shape))
v_dw2 = s_dw2 = np.zeros((w2.shape))
v_dw3 = s_dw3 = np.zeros((w3.shape))
v_dw4 = s_dw4 = np.zeros((w4.shape))
v_db1 = s_db1 = np.zeros((b1.shape))
v_db2 = s_db2 = np.zeros((b2.shape))
v_db3 = s_db3 = np.zeros((b3.shape))
v_db4 = s_db4 = np.zeros((b4.shape))


epochs = 200
lam = 0.001
alpha = 0.01
beta_1 = 0.9 # 1/(1-beta_1)
beta_2 = 0.999


batchs = 3
batch_idx = np.tile(range(batchs), int(np.ceil(len(X)/batchs)))[0:len(X)]
df = pd.DataFrame(np.concatenate((X, y[:,np.newaxis]), axis=1), index=batch_idx)

active = tanh
#active = relu
active_dv = tanh_dv
#active_dv = relu_dv
Cost_list = []

t0 = pd.Timestamp.now()
for i in range(1,epochs+1):
    Cost = 0.0
    for j in range(batchs):
        # j = 0
        X_batch = np.array(df[df.index == j])[:,0:-1]
        y_batch = np.array(df[df.index == j])[:,-1].astype(int)
        n_batch = len(X_batch)
        # 计算各输出
        z1 = X_batch.dot(w1) + b1
        a1 = active(z1)
#        a1 = drop_out(a1, keep_prob)
        z2 = a1.dot(w2) + b2
        a2 = active(z2)
#        a2 = drop_out(a2, keep_prob)
        z3 = a2.dot(w3) + b3
        a3 = active(z3)
#        a3 = drop_out(a3, keep_prob)
        z4 = a3.dot(w4) + b4
        output = softmax(z4)
        # 计算损失函数值，并判定是否跳出循环
        Loss = -np.sum(np.log(output[range(n_batch), y_batch]))/n_batch
        Loss += lam/(2*n_batch)*np.sum(w1**2) + lam/(2*n_batch)*np.sum(w2**2) + lam/(2*n_batch)*np.sum(w3**2) + lam/(2*n_batch)*np.sum(w4**2)
        Cost += Loss
        # 反向逐层求导
        delta4 = output
        delta4[range(n_batch), y_batch] -= 1
        dw4 = a3.T.dot(delta4) + lam/n_batch*w4
        db4 = np.sum(delta4, axis=0, keepdims=True)
        
        delta3 = delta4.dot(w4.T) * active_dv(z3)
        dw3 = a2.T.dot(delta3) + lam/n_batch*w3
        db3 = np.sum(delta3, axis=0, keepdims=True)
        
        delta2 = delta3.dot(w3.T) * active_dv(z2)
        dw2 = a1.T.dot(delta2) + lam/n_batch*w2
        db2 = np.sum(delta2, axis=0, keepdims=True)
        
        delta1 = delta2.dot(w2.T) * active_dv(z1)
        dw1 = X_batch.T.dot(delta1) + lam/n_batch*w1
        db1 = np.sum(delta1, axis=0, keepdims=True)
        # Adam
        w4, b4, v_dw4, v_db4, s_dw4, s_db4 = Adam(w4,b4,dw4,db4,v_dw4,v_db4,s_dw4,s_db4,beta_1,beta_2,alpha,eps,i,"no")
        w3, b3, v_dw3, v_db3, s_dw3, s_db3 = Adam(w3,b3,dw3,db3,v_dw3,v_db3,s_dw3,s_db3,beta_1,beta_2,alpha,eps,i,"no")
        w2, b2, v_dw2, v_db2, s_dw2, s_db2 = Adam(w2,b2,dw2,db2,v_dw2,v_db2,s_dw2,s_db2,beta_1,beta_2,alpha,eps,i,"no")
        w1, b1, v_dw1, v_db1, s_dw1, s_db1 = Adam(w1,b1,dw1,db1,v_dw1,v_db1,s_dw1,s_db1,beta_1,beta_2,alpha,eps,i,"no")
        # Momentum
#        w4, b4, v_dw4, v_db4 = Momentum(w4, b4, dw4, db4, v_dw4, v_db4, beta_1, alpha)
#        w3, b3, v_dw3, v_db3 = Momentum(w3, b3, dw3, db3, v_dw3, v_db3, beta_1, alpha)
#        w2, b2, v_dw2, v_db2 = Momentum(w2, b2, dw2, db2, v_dw2, v_db2, beta_1, alpha)
#        w1, b1, v_dw1, v_db1 = Momentum(w1, b1, dw1, db1, v_dw1, v_db1, beta_1, alpha)
        # gradient_descent
#        w4 = gradient_descent(alpha, w4, dw4)
#        b4 = gradient_descent(alpha, b4, db4)
#        w3 = gradient_descent(alpha, w3, dw3)
#        b3 = gradient_descent(alpha, b3, db3)
#        w2 = gradient_descent(alpha, w2, dw2)
#        b2 = gradient_descent(alpha, b2, db2)
#        w1 = gradient_descent(alpha, w1, dw1)
#        b1 = gradient_descent(alpha, b1, db1)
    # update_alpha
#    alpha = alpha_decay_2(alpha, i, 0.01)
    # Cost
    Cost /= batchs
    Cost_list.append(Cost)
    # 判定完随即储存当前最优参数
    parameter = {"w1":w1, "b1":b1,
                 "w2":w2, "b2":b2,
                 "w3":w3, "b3":b3,
                 "w4":w4, "b4":b4}
    if len(Cost_list) >= 3 and Cost_list[i-1] >= Cost_list[i-2]:
        break
t1 = pd.Timestamp.now()
print(t1-t0)

len(Cost_list)
plt.plot(Cost_list)

w1 = parameter["w1"]; b1 = parameter["b1"]
w2 = parameter["w2"]; b2 = parameter["b2"]
w3 = parameter["w3"]; b3 = parameter["b3"]
w4 = parameter["w4"]; b4 = parameter["b4"]

# predict on trainSet
z1 = X.dot(w1) + b1
a1 = active(z1)
z2 = a1.dot(w2) + b2
a2 = active(z2)
z3 = a2.dot(w3) + b3
a3 = active(z3)
z4 = a3.dot(w4) + b4
output = softmax(z4)
y_hat = np.argmax(output, axis=1)
np.sum(y_hat == y) / len(y) # 0.5139

# predict on testSet
z1 = X_predict.dot(w1) + b1
a1 = active(z1)
z2 = a1.dot(w2) + b2
a2 = active(z2)
z3 = a2.dot(w3) + b3
a3 = active(z3)
z4 = a3.dot(w4) + b4
output = softmax(z4)
y_pred = np.argmax(output, axis=1)
np.sum(y_pred == y_predict) / len(y_predict) # 0.4742 



