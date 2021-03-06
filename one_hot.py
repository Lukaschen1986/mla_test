# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from keras.utils.np_utils import to_categorical

# lambda
oht = lambda y: np.eye(len(set(y)))[y]

# map
y_dict = {"<=50K": 0, ">50K": 1}
y_oht = y.map(y_dict)

# pd.get_dummies
df = pd.get_dummies(df, columns=[], drop_first=False, dummy_na=False)
col = pd.get_dummies(col, drop_first=False, dummy_na=False)

# DataFrameMapper
mapper = DataFrameMapper(
        features=[
                ([col], OneHotEncoder()) # LabelBinarizer()
                ],
        default=False # False 全部丢弃（默认）；None 原封不动地保留
        )
mapper_fit = mapper.fit(df_train)
df_train_transform = mapper_fit.transform(df_train)
df_test_transform = mapper_fit.transform(df_test)

# keras
y_oht = to_categorical(y)

-----------------------------------------------------------------------------------

from keras.utils.np_utils import to_categorical
import pandas as pd

# one_hot
one_hot = lambda y: np.eye(len(set(y)))[y]
y_train_oht = to_categorical(y_train)
df = pd.get_dummies(df, columns=[], drop_first=False, dummy_na=False)

X,Y = ['\u4e00','\u9fa5']
X
Y
X<='jame'<=Y

df.index[0].year
df.index[0].month
df.index[0].day
df.index[0].weekday()

df[col].plot(grid=True, figsize=(10.8,7.6))

compare = pd.DataFrame({"plot_1": df[col_1],
                        "plot_2": df[col_2]})
compare.plot(grid=True, figsize=(10.8,7.6))

func = lambda x: np.max(x)-np.min(x)
df.groupby(key).transform(func)
df.groupby(key).transform("max") - df.groupby(key).transform("min")


res = pd.concat((df1,df2,df3), keys=["1","2","3"]) # 追加一列新的index
res.loc["1"]


pd.concat((df1,df2,df3), axis=1, join="inner")
pd.concat((df1,df2,df3), axis=1, join_axes=[df1.index])

df_order_full.columns = df_order_full.columns.str.replace("orderid","ordernum") # 列名修改
df_order_full.index = df_order_full.reset_index().rename(columns={"orderid","ordernum"}) # 索引名修改

df = pd.merge(df1, df2, on="...", how="inner")
df1.join(df2, how="inner")

df["col"].first_valid_index() # 列中第一个不为空的index
index > df["idx"].first_valid_index()

np.vstack((a,b,c))
np.column_stack((a,b,c))
np.trace(a)

import numpy.linalg as nplg
nplg.eig(a)
