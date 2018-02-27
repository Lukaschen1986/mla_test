# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import random as rd
from sklearn.cross_validation import train_test_split

def train_test_split(data, test_size):
    idx_test = rd.sample(range(len(data)), int(len(data)*test_size))
    idx_train = [i for i in range(len(data)) if i not in idx_test]
    return idx_train, idx_test

idx_train, idx_test = train_test_split(X, 0.1)
x_train = x[idx_train]
y_train = y[idx_train]
x_test = x[idx_test]
y_test = y[idx_test]

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=0)
