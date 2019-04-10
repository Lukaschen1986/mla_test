# -*- coding: utf-8 -*-
# https://www.jiqizhixin.com/articles/2019-02-27-6
# https://github.com/Luolc/AdaBound

import numpy as np
import pandas as pd

w = 10
vdw = 0
sdw = 0
dw = 2
beta1 = 0.9
beta2 = 0.999
eps = 10**-8
gamma = 0.001
lr = 0.001
final_lr = 0.1

df_all = pd.DataFrame()
for epoch in range(1,1001):
    # Adabound
    vdw = beta1*vdw + (1-beta1)*dw
    sdw = beta2*sdw + (1-beta2)*(dw)**2
    
    lower_bound = final_lr * (1 - 1 / (gamma*epoch+1))
    upper_bound = final_lr * (1 + 1 / (gamma*epoch))
    
    lr = np.clip(lr/(np.sqrt(sdw) + eps), lower_bound, upper_bound)
    lr = lr / np.sqrt(epoch)
    w = w - lr * vdw
    
    df = pd.DataFrame({"epoch":[epoch], 
                       "lr":[lr], 
                       "lower_bound":[lower_bound], 
                       "upper_bound":[upper_bound], 
                       "w":[w],
                       "vdw":[vdw],
                       "sdw":[sdw]})
    df_all = pd.concat([df_all,df], axis=0, ignore_index=True)
