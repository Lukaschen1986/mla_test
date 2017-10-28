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

def zscore(data, mu, sigma):
    assert isinstance(data, np.ndarray), "type of data_obj must be np.ndarray"
    res = (data-mu) / (sigma+10**-8)
    return res

# scale
X_init, X_predict_init, y, y_predict = train_test_split(data, target, test_size=0.4)
mu_init = np.mean(X_init, axis=0, keepdims=True)
sigma_init = np.std(X_init,axis=0,keepdims=True)
X = zscore(X_init, mu_init, sigma_init)
X_predict = zscore(X_predict_init, mu_init, sigma_init)
#X, X_predict, y, y_predict = train_test_split(data, target, test_size=0.4)

#def BN(data, gamma, zeta):
#    assert isinstance(data, np.ndarray), "type of data must be np.ndarray"
#    zscore = (data-np.mean(data, axis=0, keepdims=True)) / (np.std(data, axis=0, keepdims=True)+10**-8)
#    res = gamma*zscore + zeta
#    return res

def BN(data, gamma, kappa):
    assert isinstance(data, np.ndarray), "type of data must be np.ndarray"
    zscore = (data-np.mean(data, axis=0, keepdims=True)) / (np.std(data, axis=0, keepdims=True)+10**-8)
    batch_norm = gamma*zscore + kappa
    return zscore, batch_norm, np.mean(data, axis=0, keepdims=True), np.std(data, axis=0, keepdims=True)

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

def Momentum(w, dw, v_dw, beta_1, alpha, i):
    v_dw = beta_1*v_dw + (1-beta_1)*dw
    v_dw_corr = v_dw/(1-beta_1**i)
    w -= alpha * unit_vec(v_dw_corr)
    return w, v_dw_corr

def Adam(w, dw, v_dw, s_dw, beta_1, beta_2, alpha, i):
    v_dw = beta_1*v_dw + (1-beta_1)*dw
    s_dw = beta_2*s_dw + (1-beta_2)*(dw**2)
    w -= alpha * v_dw/(np.sqrt(s_dw)+10**-8)
    return w, v_dw, s_dw

#def Adam(w, dw, v_dw, s_dw, beta_1, beta_2, alpha, i):
#    v_dw = beta_1*v_dw + (1-beta_1)*dw
#    s_dw = beta_2*s_dw + (1-beta_2)*(dw**2)
#    v_dw_corr = v_dw/(1-beta_1**i)
#    s_dw_corr = s_dw/(1-beta_2**i)
#    w -= alpha * v_dw_corr/(np.sqrt(s_dw_corr)+10**-8)
#    return w, v_dw_corr, s_dw_corr

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

batchs = 1
batch_idx = np.tile(range(batchs), int(np.ceil(len(X)/batchs)))[0:len(X)]
df = pd.DataFrame(np.concatenate((X, y[:,np.newaxis]), axis=1), index=batch_idx)
print(df.index.value_counts())

#w1 = np.random.randn(p, hideNum_1) * 0.01
#w2 = np.random.randn(hideNum_1, hideNum_2) * 0.01
#w3 = np.random.randn(hideNum_2, hideNum_3) * 0.01
#w4 = np.random.randn(hideNum_3, outNum) * 0.01

np.random.seed(1)
w1 = np.random.randn(p, hideNum_1) * np.sqrt(2.0/p)
w2 = np.random.randn(hideNum_1, hideNum_2) * np.sqrt(2.0/hideNum_1)
w3 = np.random.randn(hideNum_2, hideNum_3) * np.sqrt(2.0/hideNum_2)
w4 = np.random.randn(hideNum_3, outNum) * np.sqrt(2.0/hideNum_3)
#b1 = np.zeros((1, hideNum_1))
#b2 = np.zeros((1, hideNum_2))
#b3 = np.zeros((1, hideNum_3))
#b4 = np.zeros((1, outNum))
g1 = np.ones((1, hideNum_1))
g2 = np.ones((1, hideNum_2))
g3 = np.ones((1, hideNum_3))
g4 = np.ones((1, outNum))
k1 = np.zeros((1, hideNum_1))
k2 = np.zeros((1, hideNum_2))
k3 = np.zeros((1, hideNum_3))
k4 = np.zeros((1, outNum))

v_dw1 = s_dw1 = np.zeros((w1.shape))
v_dw2 = s_dw2 = np.zeros((w2.shape))
v_dw3 = s_dw3 = np.zeros((w3.shape))
v_dw4 = s_dw4 = np.zeros((w4.shape))
#v_db1 = s_db1 = np.zeros((b1.shape))
#v_db2 = s_db2 = np.zeros((b2.shape))
#v_db3 = s_db3 = np.zeros((b3.shape))
#v_db4 = s_db4 = np.zeros((b4.shape))
v_dg1 = s_dg1 = np.zeros((g1.shape))
v_dg2 = s_dg2 = np.zeros((g2.shape))
v_dg3 = s_dg3 = np.zeros((g3.shape))
v_dg4 = s_dg4 = np.zeros((g4.shape))
v_dk1 = s_dk1 = np.zeros((k1.shape))
v_dk2 = s_dk2 = np.zeros((k2.shape))
v_dk3 = s_dk3 = np.zeros((k3.shape))
v_dk4 = s_dk4 = np.zeros((k4.shape))

v_z1_mu = v_z1_std = np.zeros((g1.shape))
v_z2_mu = v_z2_std = np.zeros((g2.shape))
v_z3_mu = v_z3_std = np.zeros((g3.shape))
v_z4_mu = v_z4_std = np.zeros((g4.shape))

epochs = 100
lam = 0.01
alpha = 0.01
beta_1 = 0.9 # 1/(1-beta_1)
beta_2 = 0.999
Cost_list = []

# alpha_decay
def alpha_decay_1(alpha, epoch, decay_rate):
    alpha_update = decay_rate**epoch * alpha
    return alpha_update

def alpha_decay_2(alpha, epoch, decay_rate):
    alpha_update = 1.0 / (1.0 + decay_rate*epoch) * alpha
    return alpha_update

#val_1 = []; val_2 = []; val_3 = []; val_4 = []
#for i in range(epochs):
#    val = alpha_decay_1(alpha, i, 0.98)
#    val_1.append(val)
#for j in range(epochs):
#    val = alpha_decay_2(alpha, j, 0.02)
#    val_2.append(val)
#alpha_decay = pd.DataFrame({"val_1": val_1, "val_2": val_2})
#alpha_decay.plot(xlim=[-2,epochs])


active = tanh
#active = relu
active_dv = tanh_dv
#active_dv = relu_dv

t0 = pd.Timestamp.now()
#i = 2
for i in range(1,epochs+1):
    Cost = 0.0
    for j in range(batchs):
        # j = 0
        X_batch = np.array(df[df.index == j])[:,0:-1]
        y_batch = np.array(df[df.index == j])[:,-1].astype(int)
        n_batch = len(X_batch)
        # 计算各层输出
        z1 = X_batch.dot(w1)
        z1_sca, z1_bn, z1_mu, z1_std = BN(z1, g1, k1)
        v_z1_mu = beta_1*v_z1_mu + (1-beta_1)*z1_mu
        v_z1_std = beta_1*v_z1_std + (1-beta_1)*z1_std
        a1 = active(z1_bn)
#        a1 = drop_out(a1, keep_prob)
        z2 = a1.dot(w2)
        z2_sca, z2_bn, z2_mu, z2_std = BN(z2, g2, k2)
        v_z2_mu = beta_1*v_z2_mu + (1-beta_1)*z2_mu
        v_z2_std = beta_1*v_z2_std + (1-beta_1)*z2_std
        a2 = active(z2_bn)
#        a2 = drop_out(a2, keep_prob)
        z3 = a2.dot(w3)
        z3_sca, z3_bn, z3_mu, z3_std = BN(z3, g3, k3)
        v_z3_mu = beta_1*v_z3_mu + (1-beta_1)*z3_mu
        v_z3_std = beta_1*v_z3_std + (1-beta_1)*z3_std
        a3 = active(z3_bn)
#        a3 = drop_out(a3, keep_prob)
        z4 = a3.dot(w4)
        z4_sca, z4_bn, z4_mu, z4_std = BN(z4, g4, k4)
        v_z4_mu = beta_1*v_z4_mu + (1-beta_1)*z4_mu
        v_z4_std = beta_1*v_z4_std + (1-beta_1)*z4_std
        output = softmax(z4_bn)
        # 计算损失函数值，并判定是否跳出循环
        Loss = -np.sum(np.log(output[range(n_batch), y_batch]))/n_batch
        Loss += lam/(2*n_batch)*np.sum(w1**2) + \
                lam/(2*n_batch)*np.sum(w2**2) + \
                lam/(2*n_batch)*np.sum(w3**2) + \
                lam/(2*n_batch)*np.sum(w4**2)
        Cost += Loss
        # 反向逐层求导
        delta4 = output
        delta4[range(n_batch), y_batch] -= 1
        dw4 = a3.T.dot(delta4)*(g4/(z4_std+10**-8)) + lam/n_batch*w4
        dg4 = np.sum(delta4.T.dot(z4_sca), axis=0, keepdims=True)
        dk4 = np.sum(delta4, axis=0, keepdims=True)
        
        delta3 = delta4.dot(w4.T) * active_dv(z3_bn)
        dw3 = a2.T.dot(delta3)*(g3/(z3_std+10**-8)) + lam/n_batch*w3
        dg3 = np.sum(delta3.T.dot(z3_sca), axis=0, keepdims=True)
        dk3 = np.sum(delta3, axis=0, keepdims=True)
        
        delta2 = delta3.dot(w3.T) * active_dv(z2_bn)
        dw2 = a1.T.dot(delta2)*(g2/(z2_std+10**-8)) + lam/n_batch*w2
        dg2 = np.sum(delta2.T.dot(z2_sca), axis=0, keepdims=True)
        dk2 = np.sum(delta2, axis=0, keepdims=True)
        
        delta1 = delta2.dot(w2.T) * active_dv(z1_bn)
        dw1 = X_batch.T.dot(delta1)*(g1/(z1_std+10**-8)) + lam/n_batch*w1
        dg1 = np.sum(delta1.T.dot(z1_sca), axis=0, keepdims=True)
        dk1 = np.sum(delta1, axis=0, keepdims=True)
        # Adam
        w4, v_dw4, s_dw4 = Adam(w4, dw4, v_dw4, s_dw4, beta_1, beta_2, alpha, i)
        g4, v_dg4, s_dg4 = Adam(g4, dg4, v_dg4, s_dg4, beta_1, beta_2, alpha, i)
        k4, v_dk4, s_dk4 = Adam(k4, dk4, v_dk4, s_dk4, beta_1, beta_2, alpha, i)
        
        w3, v_dw3, s_dw3 = Adam(w3, dw3, v_dw3, s_dw3, beta_1, beta_2, alpha, i)
        g3, v_dg3, s_dg3 = Adam(g3, dg3, v_dg3, s_dg3, beta_1, beta_2, alpha, i)
        k3, v_dk3, s_dk3 = Adam(k3, dk3, v_dk3, s_dk3, beta_1, beta_2, alpha, i)
        
        w2, v_dw2, s_dw2 = Adam(w2, dw2, v_dw2, s_dw2, beta_1, beta_2, alpha, i)
        g2, v_dg2, s_dg2 = Adam(g2, dg2, v_dg2, s_dg2, beta_1, beta_2, alpha, i)
        k2, v_dk2, s_dk2 = Adam(k2, dk2, v_dk2, s_dk2, beta_1, beta_2, alpha, i)
        
        w1, v_dw1, s_dw1 = Adam(w1, dw1, v_dw1, s_dw1, beta_1, beta_2, alpha, i)
        g1, v_dg1, s_dg1 = Adam(g1, dg1, v_dg1, s_dg1, beta_1, beta_2, alpha, i)
        k1, v_dk1, s_dk1 = Adam(k1, dk1, v_dk1, s_dk1, beta_1, beta_2, alpha, i)
        # Momentum
#        w4, v_dw4 = Momentum(w4, dw4, v_dw4, beta_1, alpha, i)
#        g4, v_dg4 = Momentum(g4, dg4, v_dg4, beta_1, alpha, i)
#        k4, v_dk4 = Momentum(k4, dk4, v_dk4, beta_1, alpha, i)
#        
#        w3, v_dw3 = Momentum(w3, dw3, v_dw3, beta_1, alpha, i)
#        g3, v_dg3 = Momentum(g3, dg3, v_dg3, beta_1, alpha, i)
#        k3, v_dk3 = Momentum(k3, dk3, v_dk3, beta_1, alpha, i)
#        
#        w2, v_dw2 = Momentum(w2, dw2, v_dw2, beta_1, alpha, i)
#        g2, v_dg2 = Momentum(g2, dg2, v_dg2, beta_1, alpha, i)
#        k2, v_dk2 = Momentum(k2, dk2, v_dk2, beta_1, alpha, i)
#        
#        w1, v_dw1 = Momentum(w1, dw1, v_dw1, beta_1, alpha, i)
#        g1, v_dg1 = Momentum(g1, dg1, v_dg1, beta_1, alpha, i)
#        k1, v_dk1 = Momentum(k1, dk1, v_dk1, beta_1, alpha, i)
        # gradient_descent
#        w4 = gradient_descent(alpha, w4, dw4)
#        g4 = gradient_descent(alpha, g4, dg4)
#        k4 = gradient_descent(alpha, k4, dk4)
#        
#        w3 = gradient_descent(alpha, w3, dw3)
#        g3 = gradient_descent(alpha, g3, dg3)
#        k3 = gradient_descent(alpha, k3, dk3)
#        
#        w2 = gradient_descent(alpha, w2, dw2)
#        g2 = gradient_descent(alpha, g2, dg2)
#        k2 = gradient_descent(alpha, k2, dk2)
#        
#        w1 = gradient_descent(alpha, w1, dw1)
#        g1 = gradient_descent(alpha, g1, dg1)
#        k1 = gradient_descent(alpha, k1, dk1)
    # update_alpha
#    alpha = alpha_decay_2(alpha, i, 0.01)
    # Cost
    Cost /= batchs
    Cost_list.append(Cost)
    # 判定完随即储存当前最优参数
    parameter = {"w1":w1, "g1":g1, "k1":k1,
                 "w2":w2, "g2":g2, "k2":k2,
                 "w3":w3, "g3":g3, "k3":k3,
                 "w4":w4, "g4":g4, "k4":k4}
    # stop condition
#    if len(Cost_list) >= 3 and Cost_list[i-1] >= Cost_list[i-2]: break
t1 = pd.Timestamp.now()
print(t1-t0)

len(Cost_list)
plt.plot(Cost_list)
np.min(Cost_list)
np.argmin(Cost_list)

#w1 = parameter["w1"]; b1 = parameter["b1"]
#w2 = parameter["w2"]; b2 = parameter["b2"]
#w3 = parameter["w3"]; b3 = parameter["b3"]
#w4 = parameter["w4"]; b4 = parameter["b4"]

w1 = parameter["w1"]; g1 = parameter["g1"]; k1 = parameter["k1"]
w2 = parameter["w2"]; g2 = parameter["g2"]; k2 = parameter["k2"]
w3 = parameter["w3"]; g3 = parameter["g3"]; k3 = parameter["k3"]
w4 = parameter["w4"]; g4 = parameter["g4"]; k4 = parameter["k4"]

# predict on trainSet
z1 = X.dot(w1)
z1_sca = zscore(z1, v_z1_mu, v_z1_std)
z1_bn = g1*z1_sca + k1
a1 = active(z1_bn)

z2 = a1.dot(w2)
z2_sca = zscore(z2, v_z2_mu, v_z2_std)
z2_bn = g2*z2_sca + k2
a2 = active(z2_bn)

z3 = a2.dot(w3)
z3_sca = zscore(z3, v_z3_mu, v_z3_std)
z3_bn = g3*z3_sca + k3
a3 = active(z3_bn)

z4 = a3.dot(w4)
z4_sca = zscore(z4, v_z4_mu, v_z4_std)
z4_bn = g4*z4_sca + k4
output = softmax(z4)
y_hat = np.argmax(output, axis=1)
np.sum(y_hat == y) / len(y) # 0.99

#z1 = X.dot(w1) + b1
#a1 = active(z1)
#z2 = a1.dot(w2) + b2
#a2 = active(z2)
#z3 = a2.dot(w3) + b3
#a3 = active(z3)
#z4 = a3.dot(w4) + b4
#output = softmax(z4)
#y_hat = np.argmax(output, axis=1)
#np.sum(y_hat == y) / len(y) # 0.9740

# predict on testSet
z1 = X_predict.dot(w1)
z1_sca = zscore(z1, v_z1_mu, v_z1_std)
z1_bn = g1*z1_sca + k1
a1 = active(z1_bn)

z2 = a1.dot(w2)
z2_sca = zscore(z2, v_z2_mu, v_z2_std)
z2_bn = g2*z2_sca + k2
a2 = active(z2_bn)

z3 = a2.dot(w3)
z3_sca = zscore(z3, v_z3_mu, v_z3_std)
z3_bn = g3*z3_sca + k3
a3 = active(z3_bn)

z4 = a3.dot(w4)
z4_sca = zscore(z4, v_z4_mu, v_z4_std)
z4_bn = g4*z4_sca + k4
output = softmax(z4)
y_pred = np.argmax(output, axis=1)
np.sum(y_pred == y_predict) / len(y_predict) # 0.9332

#z1 = X_predict.dot(w1) + b1
#a1 = active(z1)
#z2 = a1.dot(w2) + b2
#a2 = active(z2)
#z3 = a2.dot(w3) + b3
#a3 = active(z3)
#z4 = a3.dot(w4) + b4
#output = softmax(z4)
#y_pred = np.argmax(output, axis=1)
#np.sum(y_pred == y_predict) / len(y_predict) # 0.4742 

