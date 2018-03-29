# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import time
import numpy as np
import pandas as pd
import pymysql
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
from sklearn2pmml import PMMLPipeline

class VirtualQuotaRoom(object):
    def __init__(self, host, user, passwd, db):
        self.host = host
        self.user = user
        self.passwd = passwd
        self.db = db
        
    def time_sub(self, df, col_1, col_2):
        res = []
        for i in range(len(df)):
            val = (df[col_2].iloc[i]-df[col_1].iloc[i]).total_seconds()
            val = np.round(val,0)
            res.append(val)
        return res
    
    def set_ordersource(self, df):
        res = []
        for i in range(len(df)):
            if df.ordersource.iloc[i] in ("61","6"):
                val = "0"
            elif df.ordersource.iloc[i] == "62":
                val = "20"
            elif df.ordersource.iloc[i] in ("63","64"):
                val = "21"
            else:
                val = df.ordersource.iloc[i]
            res.append(val)
        return res
    
    def time_process(self, df, colname):
        month_res, month_sep_res, day_res, hour_res, hour_sep_res, weekday_res = [], [], [], [], [], []
        for data in df[colname]:
            # 1
            month = str(data.month)
            # 3
            day = str(data.day)
            # 2
            if day >= "1" and day <= "10":
                d = "1"
            elif day >= "11" and day <= "20":
                d = "2"
            else:
                d = "3"
            month_sep = month + "_" + d
            # 4
            hour = str(data.hour)
            # 5
            minute = data.minute
            minute_interval = str(np.where((minute >= 0) & (minute <= 30), "1", "2"))
            hour_sep = hour + "_" + minute_interval
            # 6
            weekday = data.weekday_name
            # append
            month_res.append(month)
            month_sep_res.append(month_sep)
            day_res.append(day)
            hour_res.append(hour)
            hour_sep_res.append(hour_sep)
            weekday_res.append(weekday)
        return month_res, month_sep_res, day_res, hour_res, hour_sep_res, weekday_res
