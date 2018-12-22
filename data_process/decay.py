# -*- coding: utf-8 -*-
# https://www.jiqizhixin.com/articles/2018-12-18-24 # 用一条数学公式破解人类记忆
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

decay_1 = lambda p,q,r,t: (p-q)*np.exp(-(p+r)*t) / (p+r-q)
decay_2 = lambda p,q,r,t: r*np.exp(-q*t) / (p+r-q)
decay = lambda p,q,r,t: ((p-q)*np.exp(-(p+r)*t) + r*np.exp(-q*t)) / (p+r-q)

p = 0.15; q = 0.005; r = 0.1
p = 0.75; q = 0.05; r = 0.03
lr = 1

lr_res = []; lr_res_1 = []; lr_res_2 = []
for t in range(100):
    val = lr * decay(p, q, r, t)
    val_1 = lr * decay_1(p, q, r, t)
    val_2 = lr * decay_2(p, q, r, t)
    lr_res.append(val)
    lr_res_1.append(val_1)
    lr_res_2.append(val_2)
   
plt.plot(lr_res)
plt.plot(lr_res_1)
plt.plot(lr_res_2)
