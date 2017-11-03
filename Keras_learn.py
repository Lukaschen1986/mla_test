# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import os
os.getcwd()
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from keras.datasets import mnist
import matplotlib.pyplot as plt

# 训练集，测试集收集非常方便
(X_init, y), (X_predict_init, y_predict) = mnist.load_data()
X_init.shape # (60000, 28, 28)

# 输入的图片是28*28像素的灰度图
img_rows, img_cols = X_init.shape[1], X_init.shape[2]

X_init = X_init.reshape(X_init.shape[0], img_rows*img_cols)
X_predict_init = X_predict_init.reshape(X_predict_init.shape[0], img_rows*img_cols)
val_max = np.max(X_init)

X = X_init / val_max
X_predict = X_predict_init / val_max

X = X.astype("float32")
X_predict = X_predict.astype("float32")
y = y.astype("int32")
y_predict = y_predict.astype("int32")

x = len(X)
p = X.shape[1]

hideNum_1 = 70
hideNum_2 = 70
hideNum_3 = 70
hideNum_4 = 70
hideNum_5 = 70
outNum = len(set(y))

batchs = 1
batch_idx = np.tile(range(batchs), int(np.ceil(len(X)/batchs)))[0:len(X)]
df = pd.DataFrame(np.concatenate((X, y[:,np.newaxis]), axis=1), index=batch_idx, dtype="float32")
print(df.index.value_counts())

np.random.seed(1)
w1 = np.random.randn(p, hideNum_1) * np.sqrt(2.0/p)
w2 = np.random.randn(hideNum_1, hideNum_2) * np.sqrt(2.0/hideNum_1)
w3 = np.random.randn(hideNum_2, hideNum_3) * np.sqrt(2.0/hideNum_2)
w4 = np.random.randn(hideNum_3, hideNum_4) * np.sqrt(2.0/hideNum_3)
w5 = np.random.randn(hideNum_4, hideNum_5) * np.sqrt(2.0/hideNum_4)
w6 = np.random.randn(hideNum_5, outNum) * np.sqrt(2.0/hideNum_5)

g1 = np.ones((1, hideNum_1))
g2 = np.ones((1, hideNum_2))
g3 = np.ones((1, hideNum_3))
g4 = np.ones((1, hideNum_4))
g5 = np.ones((1, hideNum_5))
g6 = np.ones((1, outNum))

k1 = np.zeros((1, hideNum_1))
k2 = np.zeros((1, hideNum_2))
k3 = np.zeros((1, hideNum_3))
k4 = np.zeros((1, hideNum_4))
k5 = np.zeros((1, hideNum_5))
k6 = np.zeros((1, outNum))

v_dw1 = s_dw1 = np.zeros((w1.shape))
v_dw2 = s_dw2 = np.zeros((w2.shape))
v_dw3 = s_dw3 = np.zeros((w3.shape))
v_dw4 = s_dw4 = np.zeros((w4.shape))
v_dw5 = s_dw5 = np.zeros((w5.shape))
v_dw6 = s_dw6 = np.zeros((w6.shape))

v_dg1 = s_dg1 = np.zeros((g1.shape))
v_dg2 = s_dg2 = np.zeros((g2.shape))
v_dg3 = s_dg3 = np.zeros((g3.shape))
v_dg4 = s_dg4 = np.zeros((g4.shape))
v_dg5 = s_dg5 = np.zeros((g5.shape))
v_dg6 = s_dg6 = np.zeros((g6.shape))

v_dk1 = s_dk1 = np.zeros((k1.shape))
v_dk2 = s_dk2 = np.zeros((k2.shape))
v_dk3 = s_dk3 = np.zeros((k3.shape))
v_dk4 = s_dk4 = np.zeros((k4.shape))
v_dk5 = s_dk5 = np.zeros((k5.shape))
v_dk6 = s_dk6 = np.zeros((k6.shape))

v_du1 = s_du1 = np.zeros((g1.shape))
v_du2 = s_du2 = np.zeros((g2.shape))
v_du3 = s_du3 = np.zeros((g3.shape))
v_du4 = s_du4 = np.zeros((g4.shape))
v_du5 = s_du5 = np.zeros((g5.shape))
v_du6 = s_du6 = np.zeros((g6.shape))

v_ds1 = s_ds1 = np.zeros((g1.shape))
v_ds2 = s_ds2 = np.zeros((g2.shape))
v_ds3 = s_ds3 = np.zeros((g3.shape))
v_ds4 = s_ds4 = np.zeros((g4.shape))
v_ds5 = s_ds5 = np.zeros((g5.shape))
v_ds6 = s_ds6 = np.zeros((g6.shape))

epochs = 50
lam = 0.0001
alpha = 0.001
beta_1 = 0.9 # 1/(1-beta_1)
beta_2 = 0.999
Cost_list = []; accu_train_list = []; accu_test_list = []

active = tanh
active_dv = tanh_dv

t0 = pd.Timestamp.now()
#i = 1
for i in range(1,epochs+1):
    Cost = 0.0
    for j in range(batchs):
        # j = 0
        X_batch = np.array(df[df.index == j])[:,0:-1]
        y_batch = np.array(df[df.index == j])[:,-1].astype(int)
        n_batch = len(X_batch)
        # 计算各层输出
        z1 = X_batch.dot(w1)
        z1_sca, z1_bn, u1, s1 = BN(z1, g1, k1)
        a1 = active(z1_bn)
#        a1 = drop_out(a1, keep_prob)
        z2 = a1.dot(w2)
        z2_sca, z2_bn, u2, s2 = BN(z2, g2, k2)
        a2 = active(z2_bn)
#        a2 = drop_out(a2, keep_prob)
        z3 = a2.dot(w3)
        z3_sca, z3_bn, u3, s3 = BN(z3, g3, k3)
        a3 = active(z3_bn)
#        a3 = drop_out(a3, keep_prob)
        z4 = a3.dot(w4)
        z4_sca, z4_bn, u4, s4 = BN(z4, g4, k4)
        a4 = active(z4_bn)
#        a4 = drop_out(a4, keep_prob)
        z5 = a4.dot(w5)
        z5_sca, z5_bn, u5, s5 = BN(z5, g5, k5)
        a5 = active(z5_bn)
#        a5 = drop_out(a5, keep_prob)
        z6 = a5.dot(w6)
        z6_sca, z6_bn, u6, s6 = BN(z6, g6, k6)
        output = softmax(z6_bn)
        # 计算损失函数值，并判定是否跳出循环
        Loss = -np.sum(np.log(output[range(n_batch), y_batch]))/n_batch
        Loss += lam/(2*n_batch)*np.sum(w1**2) + \
                lam/(2*n_batch)*np.sum(w2**2) + \
                lam/(2*n_batch)*np.sum(w3**2) + \
                lam/(2*n_batch)*np.sum(w4**2) + \
                lam/(2*n_batch)*np.sum(w5**2) + \
                lam/(2*n_batch)*np.sum(w6**2)
        Cost += Loss
        # 反向逐层求导
        delta6 = output
        delta6[range(n_batch), y_batch] -= 1
        dw6 = a5.T.dot(delta6)*(g6/(s6+10**-8)) + lam/n_batch*w6
        dg6 = np.sum(delta6.T.dot(z6_sca), axis=0, keepdims=True)
        dk6 = np.sum(delta6, axis=0, keepdims=True)
        du6 = -g6/(s6+10**-8)
        ds6 = np.sum(-g6*z6-g6*u6, axis=0, keepdims=True)
        
        delta5 = delta6.dot(w6.T) * active_dv(z5_bn)
        dw5 = a4.T.dot(delta5)*(g5/(s5+10**-8)) + lam/n_batch*w5
        dg5 = np.sum(delta5.T.dot(z5_sca), axis=0, keepdims=True)
        dk5 = np.sum(delta5, axis=0, keepdims=True)
        du5 = -g5/(s5+10**-8)
        ds5 = np.sum(-g5*z5-g5*u5, axis=0, keepdims=True)
                
        delta4 = delta5.dot(w5.T) * active_dv(z4_bn)
        dw4 = a3.T.dot(delta4)*(g4/(s4+10**-8)) + lam/n_batch*w4
        dg4 = np.sum(delta4.T.dot(z4_sca), axis=0, keepdims=True)
        dk4 = np.sum(delta4, axis=0, keepdims=True)
        du4 = -g4/(s4+10**-8)
        ds4 = np.sum(-g4*z4-g4*u4, axis=0, keepdims=True)
        
        delta3 = delta4.dot(w4.T) * active_dv(z3_bn)
        dw3 = a2.T.dot(delta3)*(g3/(s3+10**-8)) + lam/n_batch*w3
        dg3 = np.sum(delta3.T.dot(z3_sca), axis=0, keepdims=True)
        dk3 = np.sum(delta3, axis=0, keepdims=True)
        du3 = -g3/(s3+10**-8)
        ds3 = np.sum(-g3*z3-g3*u3, axis=0, keepdims=True)
        
        delta2 = delta3.dot(w3.T) * active_dv(z2_bn)
        dw2 = a1.T.dot(delta2)*(g2/(s2+10**-8)) + lam/n_batch*w2
        dg2 = np.sum(delta2.T.dot(z2_sca), axis=0, keepdims=True)
        dk2 = np.sum(delta2, axis=0, keepdims=True)
        du2 = -g2/(s2+10**-8)
        ds2 = np.sum(-g2*z2-g2*u2, axis=0, keepdims=True)
        
        delta1 = delta2.dot(w2.T) * active_dv(z1_bn)
        dw1 = X_batch.T.dot(delta1)*(g1/(s1+10**-8)) + lam/n_batch*w1
        dg1 = np.sum(delta1.T.dot(z1_sca), axis=0, keepdims=True)
        dk1 = np.sum(delta1, axis=0, keepdims=True)
        du1 = -g1/(s1+10**-8)
        ds1 = np.sum(-g1*z1-g1*u1, axis=0, keepdims=True)
        # Adam
        w6, v_dw6, s_dw6 = Adam(w6, dw6, v_dw6, s_dw6, beta_1, beta_2, alpha, i)
        g6, v_dg6, s_dg6 = Adam(g6, dg6, v_dg6, s_dg6, beta_1, beta_2, alpha, i)
        k6, v_dk6, s_dk6 = Adam(k6, dk6, v_dk6, s_dk6, beta_1, beta_2, alpha, i)
        u6, v_du6, s_du6 = Adam(u6, du6, v_du6, s_du6, beta_1, beta_2, alpha, i)
        s6, v_ds6, s_ds6 = Adam(s6, ds6, v_ds6, s_ds6, beta_1, beta_2, alpha, i)
        
        w5, v_dw5, s_dw5 = Adam(w5, dw5, v_dw5, s_dw5, beta_1, beta_2, alpha, i)
        g5, v_dg5, s_dg5 = Adam(g5, dg5, v_dg5, s_dg5, beta_1, beta_2, alpha, i)
        k5, v_dk5, s_dk5 = Adam(k5, dk5, v_dk5, s_dk5, beta_1, beta_2, alpha, i)
        u5, v_du5, s_du5 = Adam(u5, du5, v_du5, s_du5, beta_1, beta_2, alpha, i)
        s5, v_ds5, s_ds5 = Adam(s5, ds5, v_ds5, s_ds5, beta_1, beta_2, alpha, i)
        
        w4, v_dw4, s_dw4 = Adam(w4, dw4, v_dw4, s_dw4, beta_1, beta_2, alpha, i)
        g4, v_dg4, s_dg4 = Adam(g4, dg4, v_dg4, s_dg4, beta_1, beta_2, alpha, i)
        k4, v_dk4, s_dk4 = Adam(k4, dk4, v_dk4, s_dk4, beta_1, beta_2, alpha, i)
        u4, v_du4, s_du4 = Adam(u4, du4, v_du4, s_du4, beta_1, beta_2, alpha, i)
        s4, v_ds4, s_ds4 = Adam(s4, ds4, v_ds4, s_ds4, beta_1, beta_2, alpha, i)
        
        w3, v_dw3, s_dw3 = Adam(w3, dw3, v_dw3, s_dw3, beta_1, beta_2, alpha, i)
        g3, v_dg3, s_dg3 = Adam(g3, dg3, v_dg3, s_dg3, beta_1, beta_2, alpha, i)
        k3, v_dk3, s_dk3 = Adam(k3, dk3, v_dk3, s_dk3, beta_1, beta_2, alpha, i)
        u3, v_du3, s_du3 = Adam(u3, du3, v_du3, s_du3, beta_1, beta_2, alpha, i)
        s3, v_ds3, s_ds3 = Adam(s3, ds3, v_ds3, s_ds3, beta_1, beta_2, alpha, i)
        
        w2, v_dw2, s_dw2 = Adam(w2, dw2, v_dw2, s_dw2, beta_1, beta_2, alpha, i)
        g2, v_dg2, s_dg2 = Adam(g2, dg2, v_dg2, s_dg2, beta_1, beta_2, alpha, i)
        k2, v_dk2, s_dk2 = Adam(k2, dk2, v_dk2, s_dk2, beta_1, beta_2, alpha, i)
        u2, v_du2, s_du2 = Adam(u2, du2, v_du2, s_du2, beta_1, beta_2, alpha, i)
        s2, v_ds2, s_ds2 = Adam(s2, ds2, v_ds2, s_ds2, beta_1, beta_2, alpha, i)
        
        w1, v_dw1, s_dw1 = Adam(w1, dw1, v_dw1, s_dw1, beta_1, beta_2, alpha, i)
        g1, v_dg1, s_dg1 = Adam(g1, dg1, v_dg1, s_dg1, beta_1, beta_2, alpha, i)
        k1, v_dk1, s_dk1 = Adam(k1, dk1, v_dk1, s_dk1, beta_1, beta_2, alpha, i)
        u1, v_du1, s_du1 = Adam(u1, du1, v_du1, s_du1, beta_1, beta_1, alpha, i)
        s1, v_ds1, s_ds1 = Adam(s1, ds1, v_ds1, s_ds1, beta_1, beta_1, alpha, i)
    # update_alpha
#    alpha = alpha_decay_2(alpha, i, 0.01)
    # Cost
    Cost /= batchs
    Cost_list.append(Cost)
    # 判定完随即储存当前最优参数
    parameter = {"w1":w1, "g1":g1, "k1":k1, "u1":u1, "s1":s1,
                 "w2":w2, "g2":g2, "k2":k2, "u2":u2, "s2":s2,
                 "w3":w3, "g3":g3, "k3":k3, "u3":u3, "s3":s3,
                 "w4":w4, "g4":g4, "k4":k4, "u4":u4, "s4":s4,
                 "w5":w5, "g5":g5, "k5":k5, "u5":u5, "s5":s5,
                 "w6":w6, "g6":g6, "k6":k6, "u6":u6, "s6":s6}
    w1 = parameter["w1"]; g1 = parameter["g1"]; k1 = parameter["k1"]; u1 = parameter["u1"]; s1 = parameter["s1"]
    w2 = parameter["w2"]; g2 = parameter["g2"]; k2 = parameter["k2"]; u2 = parameter["u2"]; s2 = parameter["s2"]
    w3 = parameter["w3"]; g3 = parameter["g3"]; k3 = parameter["k3"]; u3 = parameter["u3"]; s3 = parameter["s3"]
    w4 = parameter["w4"]; g4 = parameter["g4"]; k4 = parameter["k4"]; u4 = parameter["u4"]; s4 = parameter["s4"]
    w5 = parameter["w5"]; g5 = parameter["g5"]; k5 = parameter["k5"]; u5 = parameter["u5"]; s5 = parameter["s5"]
    w6 = parameter["w6"]; g6 = parameter["g6"]; k6 = parameter["k6"]; u6 = parameter["u6"]; s6 = parameter["s6"]
    # predict_on_train
    z1 = X.dot(w1)
    z1_sca = zscore(z1, u1, s1)
    z1_bn = g1*z1_sca + k1
    a1 = active(z1_bn)
    
    z2 = a1.dot(w2)
    z2_sca = zscore(z2, u2, s2)
    z2_bn = g2*z2_sca + k2
    a2 = active(z2_bn)
    
    z3 = a2.dot(w3)
    z3_sca = zscore(z3, u3, s3)
    z3_bn = g3*z3_sca + k3
    a3 = active(z3_bn)
    
    z4 = a3.dot(w4)
    z4_sca = zscore(z4, u4, s4)
    z4_bn = g4*z4_sca + k4
    a4 = active(z4_bn)
    
    z5 = a4.dot(w5)
    z5_sca = zscore(z5, u5, s5)
    z5_bn = g5*z5_sca + k5
    a5 = active(z5_bn)
    
    z6 = a5.dot(w6)
    z6_sca = zscore(z6, u6, s6)
    z6_bn = g6*z6_sca + k6
    output = softmax(z6_bn)
    y_hat = np.argmax(output, axis=1)
    accu_train = np.sum(y_hat == y)/len(y)
    accu_train_list.append(accu_train)
    # predict_on_test
    z1 = X_predict.dot(w1)
    z1_sca = zscore(z1, u1, s1)
    z1_bn = g1*z1_sca + k1
    a1 = active(z1_bn)
    
    z2 = a1.dot(w2)
    z2_sca = zscore(z2, u2, s2)
    z2_bn = g2*z2_sca + k2
    a2 = active(z2_bn)
    
    z3 = a2.dot(w3)
    z3_sca = zscore(z3, u3, s3)
    z3_bn = g3*z3_sca + k3
    a3 = active(z3_bn)
    
    z4 = a3.dot(w4)
    z4_sca = zscore(z4, u4, s4)
    z4_bn = g4*z4_sca + k4
    a4 = active(z4_bn)
    
    z5 = a4.dot(w5)
    z5_sca = zscore(z5, u5, s5)
    z5_bn = g5*z5_sca + k5
    a5 = active(z5_bn)
    
    z6 = a5.dot(w6)
    z6_sca = zscore(z6, u6, s6)
    z6_bn = g6*z6_sca + k6
    output = softmax(z6_bn)
    y_pred = np.argmax(output, axis=1)
    accu_test = np.sum(y_pred == y_predict)/len(y_predict)
    accu_test_list.append(accu_test)
    print("epoch:",i,"Cost:",round(Cost,3),"alpha:",round(alpha,4),"accu_train:",round(accu_train,4),"accu_test:",round(accu_test,4))
    # stop condition
#    if len(Cost_list) >= 3 and Cost_list[i-1] >= Cost_list[i-2]: break
t1 = pd.Timestamp.now()
print(t1-t0)
plt.plot(Cost_list)
plt.plot(accu_train_list)
plt.plot(accu_test_list)
