# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import os
os.chdir("/home/bimining/project/itravel/virtual_quota_room/script")
import class_vqr as vqr
import pickle
import pandas as pd
import xgboost as xgb
from sklearn2pmml import sklearn2pmml

VQR = vqr.VirtualQuotaRoom(host="", user="", passwd="", db="")
# data_load
dataSet = VQR.data_load(sql=train_sql)
# data_transform
df_train, df_train_transform, df_test_transform, pre_hours, room_nights, price, mapper_fit = VQR.data_transform(df=dataSet, test_sizes=0.15)
# data_split
x_train, y_train = VQR.data_split(df_train_transform)
x_test, y_test = VQR.data_split(df_test_transform)
# train
model_res, clf = VQR.train(x_train, y_train, x_test, y_test, 
                           n_estimators_0=200, nthread=4, objective="binary:logistic", 
                           eval_metric="auc", scoring="roc_auc", nfold=5)
model_res.save_model(fname="./model_res.model")
model_res.dump_model(fout="./model_res_txt.txt", with_stats=False)
f = open("./clf.txt", "wb")
pickle.dump(clf, f); f.close()
# test
model_res = xgb.Booster({"nthread": 4})
model_res.load_model(fname="/home/bimining/project/itravel/virtual_quota_room/txt/model_res.model")
t, recall = VQR.test(x_test, y_test, model_res, proba=0.995)
# data_pmml
pipeline = VQR.data_pmml(df_train, clf)
pickle.dump(pipeline, f); f.close()
sklearn2pmml(pipeline, 
             pmml="/home/bimining/project/itravel/virtual_quota_room/txt/XGBClassifier.pmml",
             with_repr=True, # 是否打印模型参数到pmml
             debug=False)
             
