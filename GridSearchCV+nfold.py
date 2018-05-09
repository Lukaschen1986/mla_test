# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import random as rd
from sklearn.utils import shuffle
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedKFold, ShuffleSplit, StratifiedShuffleSplit

# train_test_split
## manual
def train_test_split(data, test_size):
    idx_test = rd.sample(range(len(data)), int(len(data)*test_size))
    idx_train = [i for i in range(len(data)) if i not in idx_test]
    return idx_train, idx_test

idx_train, idx_test = train_test_split(X, 0.1)
x_train = x[idx_train]
y_train = y[idx_train]
x_test = x[idx_test]
y_test = y[idx_test]
## sklearn
df = shuffle(df, n_samples=None, random_state=0)
x = df.iloc[:,0:2]
y = df.iloc[:,2]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)

# GridSearchCV + nfold
nfold = 5
nfold = KFold(n_splits=5, shuffle=True, random_state=0)
nfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0) # 首选
nfold = ShuffleSplit(n_splits=5, test_size="default", train_size=None, random_state=0)
nfold = StratifiedShuffleSplit(n_splits=5, test_size="default", train_size=None, random_state=0)

param = {"max_depth": np.arange(3,11,2), "min_child_weight": np.arange(1,6,2)}
clf_gscv = GridSearchCV(estimator=clf, param_grid=param, scoring="roc_auc", n_jobs=-1, iid=False, cv=nfold)
