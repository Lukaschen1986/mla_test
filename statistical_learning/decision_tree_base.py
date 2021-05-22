# -*- coding: utf-8 -*-
"""
Created on Sat May 22 10:47:55 2021

@author: ASUS

对比信息增益和信息增益率的差异
"""
import os
import sys
from collections import Counter
import numpy as np
import pandas as pd

path_project = os.path.dirname(os.path.dirname(__file__))
xls = pd.ExcelFile(os.path.join(path_project, "data/test.xlsx"))
df = xls.parse(sheet_name=xls.sheet_names[0])


def entropy(df, col):
    """
    col = "speed"
    """
    x = df[col]
    n = len(x)
    count = Counter(x)
    H = 0
    
    for (_,v) in count.items():
        H += v/n * np.log2(v/n)
        
    return -H
    

def conditional_entropy(df, col, target):
    """
    col = "accelerate"
    target = "speed"
    """
    x = df[col]
    n = len(x)
    count = Counter(x)
    HC = 0
    
    for (k,v) in count.items():
        df_sub = df[df[col] == k]
        HC += v/n * entropy(df_sub, target)
        
    return HC

info_gain = lambda H, HC: H - HC
info_gain_rate = lambda IG, H: IG / H


if __name__ == "__main__":
    print(path_project)
    
    H_y = entropy(df, col="speed")
    
    HC_x = conditional_entropy(df, col="limit", target="speed")
    IG_x = info_gain(H_y, HC_x)
    H_x = entropy(df, col="limit")
    IGR_x = info_gain_rate(IG_x, H_x)

    HC_x = conditional_entropy(df, col="grade", target="speed")
    IG_x = info_gain(H_y, HC_x)
    H_x = entropy(df, col="grade")
    IGR_x = info_gain_rate(IG_x, H_x)

    HC_x = conditional_entropy(df, col="accelerate", target="speed")
    IG_x = info_gain(H_y, HC_x)
    H_x = entropy(df, col="accelerate")
    IGR_x = info_gain_rate(IG_x, H_x)
