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
df.dtypes
print(df.index.value_counts())

np.random.seed(1)
w1 = np.random.randn(p, hideNum_1) * np.sqrt(2.0/p)
w2 = np.random.randn(hideNum_1, hideNum_2) * np.sqrt(2.0/hideNum_1)
w3 = np.random.randn(hideNum_2, hideNum_3) * np.sqrt(2.0/hideNum_2)
w4 = np.random.randn(hideNum_3, hideNum_4) * np.sqrt(2.0/hideNum_3)
w5 = np.random.randn(hideNum_4, hideNum_5) * np.sqrt(2.0/hideNum_4)
w6 = np.random.randn(hideNum_5, outNum) * np.sqrt(2.0/hideNum_5)

b1 = np.zeros((1, hideNum_1))
b2 = np.zeros((1, hideNum_2))
b3 = np.zeros((1, hideNum_3))
b4 = np.zeros((1, hideNum_4))
b5 = np.zeros((1, hideNum_5))
b6 = np.zeros((1, outNum))

v_dw1 = s_dw1 = np.zeros((w1.shape))
v_dw2 = s_dw2 = np.zeros((w2.shape))
v_dw3 = s_dw3 = np.zeros((w3.shape))
v_dw4 = s_dw4 = np.zeros((w4.shape))
v_dw5 = s_dw5 = np.zeros((w5.shape))
v_dw6 = s_dw6 = np.zeros((w6.shape))

v_db1 = s_db1 = np.zeros((b1.shape))
v_db2 = s_db2 = np.zeros((b2.shape))
v_db3 = s_db3 = np.zeros((b3.shape))
v_db4 = s_db4 = np.zeros((b4.shape))
v_db5 = s_db5 = np.zeros((b5.shape))
v_db6 = s_db6 = np.zeros((b6.shape))

epochs = 50
lam = 0.0001
alpha = 0.01
beta_1 = 0.9 # 1/(1-beta_1)
beta_2 = 0.999
Cost_list = []

active = relu
active_dv = relu_dv

t0 = pd.Timestamp.now()
#i = 1
for i in range(1,epochs+1):
    Cost = 0.0
    for j in range(batchs):
        # j = 0
        X_batch = np.array(df[df.index == j])[:,0:-1]
        y_batch = np.array(df[df.index == j])[:,-1].astype("int32")
        n_batch = len(X_batch)
        # 计算各层输出
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
        a4 = active(z4)
#        a4 = drop_out(a4, keep_prob)
        z5 = a4.dot(w5) + b5
        a5 = active(z5)
#        a5 = drop_out(a5, keep_prob)
        z6 = a5.dot(w6) + b6
        output = softmax(z6)
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
        dw6 = a5.T.dot(delta6) + lam/n_batch*w6
        db6 = np.sum(delta6, axis=0, keepdims=True)
        
        delta5 = delta6.dot(w6.T) * active_dv(z5)
        dw5 = a4.T.dot(delta5) + lam/n_batch*w5
        db5 = np.sum(delta5, axis=0, keepdims=True)
                
        delta4 = delta5.dot(w5.T) * active_dv(z4)
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
        w6, v_dw6, s_dw6 = Adam(w6, dw6, v_dw6, s_dw6, beta_1, beta_2, alpha, i)
        b6, v_db6, s_db6 = Adam(b6, db6, v_db6, s_db6, beta_1, beta_2, alpha, i)
        
        w5, v_dw5, s_dw5 = Adam(w5, dw5, v_dw5, s_dw5, beta_1, beta_2, alpha, i)
        b5, v_db5, s_db5 = Adam(b5, db5, v_db5, s_db5, beta_1, beta_2, alpha, i)
        
        w4, v_dw4, s_dw4 = Adam(w4, dw4, v_dw4, s_dw4, beta_1, beta_2, alpha, i)
        b4, v_db4, s_db4 = Adam(b4, db4, v_db4, s_db4, beta_1, beta_2, alpha, i)
        
        w3, v_dw3, s_dw3 = Adam(w3, dw3, v_dw3, s_dw3, beta_1, beta_2, alpha, i)
        b3, v_db3, s_db3 = Adam(b3, db3, v_db3, s_db3, beta_1, beta_2, alpha, i)
        
        w2, v_dw2, s_dw2 = Adam(w2, dw2, v_dw2, s_dw2, beta_1, beta_2, alpha, i)
        b2, v_db2, s_db2 = Adam(b2, db2, v_db2, s_db2, beta_1, beta_2, alpha, i)
        
        w1, v_dw1, s_dw1 = Adam(w1, dw1, v_dw1, s_dw1, beta_1, beta_2, alpha, i)
        b1, v_db1, s_db1 = Adam(b1, db1, v_db1, s_db1, beta_1, beta_2, alpha, i)

    # update_alpha
#    alpha = alpha_decay_2(alpha, i, 0.01)
    # Cost
    Cost /= batchs
    Cost_list.append(Cost)
    # 判定完随即储存当前最优参数
    parameter = {"w1":w1, "b1":b1,
                 "w2":w2, "b2":b2,
                 "w3":w3, "b3":b3,
                 "w4":w4, "b4":b4,
                 "w5":w5, "b5":b5,
                 "w6":w6, "b6":b6}
    # predict_on_train
    z1 = X.dot(w1) + b1
    a1 = active(z1)
    
    z2 = a1.dot(w2) + b2
    a2 = active(z2)
    
    z3 = a2.dot(w3) + b3
    a3 = active(z3)
    
    z4 = a3.dot(w4) + b4
    a4 = active(z4)
    
    z5 = a4.dot(w5) + b5
    a5 = active(z5)
    
    z6 = a5.dot(w6) + b6
    output = softmax(z6)
    y_hat = np.argmax(output, axis=1)
    accu_train = np.sum(y_hat == y)/len(y)
    # predict_on_test
    z1 = X_predict.dot(w1) + b1
    a1 = active(z1)
    
    z2 = a1.dot(w2) + b2
    a2 = active(z2)
    
    z3 = a2.dot(w3) + b3
    a3 = active(z3)
    
    z4 = a3.dot(w4) + b4
    a4 = active(z4)
    
    z5 = a4.dot(w5) + b5
    a5 = active(z5)
    
    z6 = a5.dot(w6) + b6
    output = softmax(z6)
    y_pred = np.argmax(output, axis=1)
    accu_test = np.sum(y_pred == y_predict)/len(y_predict)
    print("epoch:",i,"Cost:",round(Cost,3),"accu_train:",round(accu_train,4),"accu_test:",round(accu_test,4))
    # stop condition
#    if len(Cost_list) >= 3 and Cost_list[i-1] >= Cost_list[i-2]: break
t1 = pd.Timestamp.now()
print(t1-t0)
plt.plot(Cost_list)
