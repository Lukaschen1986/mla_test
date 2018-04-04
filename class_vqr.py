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
    
    def data_load(self, sql):
        itravelhw = pymysql.connect(self.host, self.user, self.passwd, self.db, charset="utf8")
        itravelhw.commit()
    
        t0 = pd.Timestamp.now()
        dataSet = pd.read_sql(sql=sql, con=itravelhw)
        t1 = pd.Timestamp.now()
        itravelhw.close()
    
        dataSet.columns = dataSet.columns.str.lower()
        print("数据提取完毕，用时：%s" % (t1-t0))
        return dataSet
    
    def data_transform(self, df, test_sizes):
        t0 = pd.Timestamp.now()
        # 1-划分数据集，形成常规房训练集
        df_normal_confirm = df[df.orderstatus.isin(["3","11"])]
        df_normal_cancel_ctm = df[df.cancelreasonid.isin(["2","201","202","203","204","3","4","7"])]
        df_normal_cancel_aft = df_normal_cancel_ctm[-pd.isnull(df_normal_cancel_ctm.confirmtime)]
        df_normal_useful = pd.concat((df_normal_confirm, df_normal_cancel_aft), axis=0)
        df_normal_useless = df[df.cancelreasonid.isin(["1","5","8"]) | -pd.isnull(df.cancelreasonidother)]
        df_normal_useful["is_confirm"] = 1
        df_normal_useless["is_confirm"] = 0
        df_normal = pd.concat((df_normal_useful, df_normal_useless), axis=0, ignore_index=True)
        # 2-计算确认时长（分钟），基于下单时间
        df_normal["confirmseconds"] = self.time_sub(df_normal, "createtime", "confirmtime")
        df_normal = df_normal[-(df_normal.confirmseconds < 0)] # 剔除confirmseconds为负的数据
        # 3-计算回传时长（分钟），基于发单时间
        df_normal["returnseconds"] = self.time_sub(df_normal, "to_hotel", "confirmtime")
        df_normal = df_normal[-(df_normal.returnseconds < 0)] # 剔除returnseconds为负的数据
        # 4-计算提前预定时长（小时）并离散化
        df_normal["pre_seconds"] = self.time_sub(df_normal, "createtime", "latestarrivaltime")
        df_normal["pre_hours"] = df_normal.pre_seconds / 3600 # 计算提前预定时长（小时）
        df_normal = df_normal.drop("pre_seconds", axis=1)
        
        pre_hours_q1, pre_hours_q2, pre_hours_q3, pre_hours_q4, pre_hours_q5, pre_hours_q6 = np.round(np.percentile(df_normal.pre_hours[df_normal.pre_hours > 0], q=[5,10,20,40,60,80])).astype(np.int32)
        pre_hours = {"pre_hours_q1": pre_hours_q1,
                     "pre_hours_q2": pre_hours_q2,
                     "pre_hours_q3": pre_hours_q3,
                     "pre_hours_q4": pre_hours_q4,
                     "pre_hours_q5": pre_hours_q5,
                     "pre_hours_q6": pre_hours_q6}
        
        df_normal["pre_hours_disperse"] = np.nan
        df_normal.pre_hours_disperse[df_normal.pre_hours <= pre_hours_q1] = "(-inf_%d)" % pre_hours_q1
        df_normal.pre_hours_disperse[(df_normal.pre_hours > pre_hours_q1) & (df_normal.pre_hours <= pre_hours_q2)] = "(%d_%d)" % (pre_hours_q1, pre_hours_q2)
        df_normal.pre_hours_disperse[(df_normal.pre_hours > pre_hours_q2) & (df_normal.pre_hours <= pre_hours_q3)] = "(%d_%d)" % (pre_hours_q2, pre_hours_q3)
        df_normal.pre_hours_disperse[(df_normal.pre_hours > pre_hours_q3) & (df_normal.pre_hours <= pre_hours_q4)] = "(%d_%d)" % (pre_hours_q3, pre_hours_q4)
        df_normal.pre_hours_disperse[(df_normal.pre_hours > pre_hours_q4) & (df_normal.pre_hours <= pre_hours_q5)] = "(%d_%d)" % (pre_hours_q4, pre_hours_q5)
        df_normal.pre_hours_disperse[(df_normal.pre_hours > pre_hours_q5) & (df_normal.pre_hours <= pre_hours_q6)] = "(%d_%d)" % (pre_hours_q5, pre_hours_q6)
        df_normal.pre_hours_disperse[df_normal.pre_hours > pre_hours_q6] = "(%d_inf)" % pre_hours_q6
        
        # 5-计算是否晚于最后到店时间预定
        df_normal["create_late"] = np.where(df_normal.pre_hours < 0, 1, 0)
        # 6-计算是否是预定当天
        df_normal["create_sameday"] = np.where(df_normal.pre_hours <= 24, 1, 0)
        # 7-计算是否是预定两天以内
        df_normal["create_twoday"] = np.where(df_normal.pre_hours <= 48, 1, 0)
        # 8-计算入住间夜并离散化
        df_normal["nights"] = self.time_sub(df_normal, "checkindate", "checkoutdate")
        df_normal.nights /= 86400
        df_normal["room_nights"] = df_normal.checkinnumofrooms * df_normal.nights
        df_normal = df_normal.drop("nights", axis=1)
        
        room_nights_q1, room_nights_q2, room_nights_q3, room_nights_q4 = np.round(np.percentile(df_normal.room_nights, q=[20,40,60,80])).astype(np.int32)
        room_nights = {"room_nights_q1": room_nights_q1,
                       "room_nights_q2": room_nights_q2,
                       "room_nights_q3": room_nights_q3,
                       "room_nights_q4": room_nights_q4}
        
        df_normal["room_nights_disperse"] = np.nan
        df_normal.room_nights_disperse[df_normal.room_nights <= room_nights_q1] = "(0_%d)" % room_nights_q1
        df_normal.room_nights_disperse[(df_normal.room_nights > room_nights_q1) & (df_normal.room_nights <= room_nights_q2)] = "(%d_%d)" % (room_nights_q1, room_nights_q2)
        df_normal.room_nights_disperse[(df_normal.room_nights > room_nights_q2) & (df_normal.room_nights <= room_nights_q3)] = "(%d_%d)" % (room_nights_q2, room_nights_q3)
        df_normal.room_nights_disperse[(df_normal.room_nights > room_nights_q3) & (df_normal.room_nights <= room_nights_q4)] = "(%d_%d)" % (room_nights_q3, room_nights_q4)
        df_normal.room_nights_disperse[df_normal.room_nights > room_nights_q4] = "(%d_inf)" % room_nights_q4
        # 9-计算房费金额并离散化
        price_q1, price_q2, price_q3, price_q4 = np.round(np.percentile(df_normal.contractprice, q=[20,40,60,80])).astype(np.int32)
        price = {"price_q1": price_q1,
                 "price_q2": price_q2,
                 "price_q3": price_q3,
                 "price_q4": price_q4}
        
        df_normal["contractprice_disperse"] = np.nan
        df_normal.contractprice_disperse[df_normal.contractprice <= price_q1] = "(0_%d)" % price_q1
        df_normal.contractprice_disperse[(df_normal.contractprice > price_q1) & (df_normal.contractprice <= price_q2)] = "(%d_%d)" % (price_q1, price_q2)
        df_normal.contractprice_disperse[(df_normal.contractprice > price_q2) & (df_normal.contractprice <= price_q3)] = "(%d_%d)" % (price_q2, price_q3)
        df_normal.contractprice_disperse[(df_normal.contractprice > price_q3) & (df_normal.contractprice <= price_q4)] = "(%d_%d)" % (price_q3, price_q4)
        df_normal.contractprice_disperse[df_normal.contractprice > price_q4] = "(%d_inf)" % price_q4
        # 10-整理钻级，为空置为99
        df_normal.hotelstargrade = np.where(df_normal.hotelstargrade.isin([""]), "99", df_normal.hotelstargrade)
        # 11-整理渠道
        df_normal = df_normal[-df_normal.ordersource.isin(["22","23"])] # 剔除TravelOne（IOS），TravelOne（Android）渠道
        df_normal.ordersource = self.set_ordersource(df_normal)
        # 12-支付方式
        df_normal = df_normal[df_normal.payway != ""] # 剔除空白单
        # 13-下单时间处理
        _, _create_month_sep, _create_day, _, _create_hour_sep, _create_weekday = self.time_process(df_normal, "createtime")
        df_normal["create_month"] = _create_month_sep
        df_normal["create_day"] = _create_day
        df_normal["create_hour"] = _create_hour_sep
        df_normal["create_weekday"] = _create_weekday
        # 14-发单时间处理
        _, _send_month_sep, _send_day, _, _send_hour_sep, _send_weekday = self.time_process(df_normal, "to_hotel")
        df_normal["send_month"] = _send_month_sep
        df_normal["send_day"] = _send_day
        df_normal["send_hour"] = _send_hour_sep
        df_normal["send_weekday"] = _send_weekday
        # 15-入住时间处理
        _, _checkin_month_sep, _checkin_day, _, _, _checkin_weekday = self.time_process(df_normal, "checkindate")
        df_normal["checkin_month"] = _checkin_month_sep
        df_normal["checkin_day"] = _checkin_day
        df_normal["checkin_weekday"] = _checkin_weekday
        # 16-筛选建模特征
        colnames = ["is_confirm","orderattribute","ordersource",\
                    "payway","cityid","hotelstargrade","contractprice","contractprice_disperse",\
                    "guaranstatus","create_late","create_sameday","create_twoday","pre_hours","pre_hours_disperse",\
                    "room_nights","room_nights_disperse","create_month","create_day","create_hour",\
                    "create_weekday","checkin_month","checkin_day","checkin_weekday"]
        df_model = df_normal[colnames]
        
        df_model.orderattribute = df_model.orderattribute.astype(np.int64)
        df_model.ordersource = df_model.ordersource.astype(np.int64)
        df_model.payway = df_model.payway.astype(np.int64)
        df_model.hotelstargrade = df_model.hotelstargrade.astype(np.int64)
        df_model.guaranstatus = df_model.guaranstatus.astype(np.int64)
        df_model.create_late = df_model.create_late.astype(np.int64)
        df_model.create_sameday = df_model.create_sameday.astype(np.int64)
        df_model.create_twoday = df_model.create_twoday.astype(np.int64)
        df_model.create_day = df_model.create_day.astype(np.int64)
        df_model.checkin_day = df_model.checkin_day.astype(np.int64)
        # 17-train_test_split
        df_model = shuffle(df_model, random_state=0)
        df_train, df_test = train_test_split(df_model, test_size=test_sizes, random_state=0)
        # 18-特征变换
        mapper = DataFrameMapper(
            features=[
                    (["is_confirm"], None),
                    (["contractprice"], None),
                    (["pre_hours"], None),
                    (["room_nights"], None),
                    (["orderattribute"], OneHotEncoder()),
                    (["ordersource"], OneHotEncoder()),
                    (["payway"], OneHotEncoder()),
                    (["cityid"], LabelBinarizer()),
                    (["hotelstargrade"], OneHotEncoder()),
                    (["contractprice_disperse"], LabelBinarizer()),
                    (["guaranstatus"], OneHotEncoder()),
                    (["create_late"], OneHotEncoder()),
                    (["create_sameday"], OneHotEncoder()),
                    (["create_twoday"], OneHotEncoder()),
                    (["pre_hours_disperse"], LabelBinarizer()),
                    (["room_nights_disperse"], LabelBinarizer()),
                    (["create_month"], LabelBinarizer()),
                    (["create_day"], OneHotEncoder()),
                    (["create_hour"], LabelBinarizer()),
                    (["create_weekday"], LabelBinarizer()),
                    (["checkin_month"], LabelBinarizer()),
                    (["checkin_day"], OneHotEncoder()),
                    (["checkin_weekday"], LabelBinarizer())
                    ],
            default=False
            )
        mapper_fit = mapper.fit(df_train)
        df_train_transform = mapper_fit.transform(df_train)
        df_test_transform = mapper_fit.transform(df_test)
        # 19-特征名称
        df_train_transform_2 = pd.get_dummies(df_train, columns=["orderattribute","ordersource","payway","cityid","hotelstargrade",
                                                                 "contractprice_disperse","guaranstatus","create_late","create_sameday",
                                                                 "create_twoday","pre_hours_disperse","room_nights_disperse","create_month",
                                                                 "create_day","create_hour","create_weekday","checkin_month",
                                                                 "checkin_day","checkin_weekday"], drop_first=False, dummy_na=False)
        feat_name = df_train_transform_2.columns[1:]
        t1 = pd.Timestamp.now()
        print("特征工程处理完毕，用时：%s" % (t1-t0))
        return df_train, df_train_transform, df_test_transform, pre_hours, room_nights, price, mapper_fit, feat_name
    
    def data_split(self, df_model_transform):
        x = df_model_transform[:,1:]
        y = df_model_transform[:,0]
        return x, y
    
    def train(self, x_train, y_train, x_test, y_test, n_estimators_0, nthread, objective, eval_metric, scoring, nfold):
        # 1-设置参数初始值
        print("1-设置参数初始值")
        clf = XGBClassifier(
            # General Parameters
            booster="gbtree",
            silent=1,
            nthread=nthread,
            # Booster Parameters
            learning_rate=0.3,
            n_estimators=n_estimators_0,
            gamma=0,
            max_depth=6,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0,
            reg_lambda=1,
            max_delta_step=0,
            scale_pos_weight=1,
            # Learning Task Parameters
            objective=objective,
            eval_metric=eval_metric,
            #num_class=len(set(y_train)),
            seed=0
            )
        
        # 2-训练最优弱分类器个数：n_estimators_1
        print("2-训练最优弱分类器个数：n_estimators_1")
        xgb_param = clf.get_xgb_params()
        d_train = xgb.DMatrix(x_train, y_train)
        d_test = xgb.DMatrix(x_test, y_test)
        watchlist = [(d_train, "train"), (d_test, "test")]
        
        t_begin = pd.Timestamp.now()
        xgb_cv = xgb.cv(params=xgb_param, 
                        dtrain=d_train, 
                        num_boost_round=xgb_param["n_estimators"],
                        nfold=nfold, 
                        metrics=eval_metric, 
                        early_stopping_rounds=int(xgb_param["n_estimators"]/10),
                        verbose_eval=None)
        t1 = pd.Timestamp.now()
        n_estimators_1 = xgb_cv.shape[0]
        clf.set_params(n_estimators=n_estimators_1)
        xgb_param = clf.get_xgb_params()
        print("分类器个数：%s， 用时：%s" % (n_estimators_1, (t1-t_begin)))
        time.sleep(30)
        
        # 3-暴力搜索：learning_rate
        print("3-暴力搜索：learning_rate")
        param = {"learning_rate": np.arange(0.1, 0.6, 0.1)}
        clf_gscv = GridSearchCV(estimator=clf, param_grid=param, scoring=scoring, n_jobs=nthread, iid=False, cv=nfold)
        
        t0 = pd.Timestamp.now()
        model_3 = clf_gscv.fit(x_train, y_train)
        t1 = pd.Timestamp.now()
        #model_3.grid_scores_; model_3.best_score_
        best_param = model_3.best_params_["learning_rate"]
        clf.set_params(learning_rate=best_param)
        xgb_param = clf.get_xgb_params()
        print("learning_rate：%s， 用时：%s" % (best_param, (t1-t0)))
        time.sleep(60)
        
        # 4-暴力搜索：max_depth, min_child_weight
        print("4-暴力搜索：max_depth, min_child_weight")
        param = {"max_depth": np.arange(3,11,2), "min_child_weight": np.arange(1,6,2)}
        clf_gscv = GridSearchCV(estimator=clf, param_grid=param, scoring=scoring, n_jobs=nthread, iid=False, cv=nfold)
        
        t0 = pd.Timestamp.now()
        model_4 = clf_gscv.fit(x_train, y_train)
        t1 = pd.Timestamp.now()
        best_param_1 = model_4.best_params_["max_depth"]
        best_param_2 = model_4.best_params_["min_child_weight"]
        print("max_depth：%s，min_child_weight：%s，用时：%s" % (best_param_1, best_param_2, (t1-t0)))
        time.sleep(60)
        
        # 5-精确搜索：max_depth, min_child_weight
        print("5-精确搜索：max_depth, min_child_weight")
        param = {"max_depth": [best_param_1-1, best_param_1, best_param_1+1], 
                 "min_child_weight": [best_param_2-1, best_param_2, best_param_2+1]}
        clf_gscv = GridSearchCV(estimator=clf, param_grid=param, scoring=scoring, n_jobs=nthread, iid=False, cv=nfold)
        
        t0 = pd.Timestamp.now()
        model_5 = clf_gscv.fit(x_train, y_train)
        t1 = pd.Timestamp.now()
        best_param_1 = model_5.best_params_["max_depth"]
        best_param_2 = model_5.best_params_["min_child_weight"]
        clf.set_params(max_depth=best_param_1, min_child_weight=best_param_2)
        xgb_param = clf.get_xgb_params()
        print("max_depth：%s，min_child_weight：%s，用时：%s" % (best_param_1, best_param_2, (t1-t0)))
        time.sleep(60)
        
        # 6-暴力搜索：gamma
        print("6-暴力搜索：gamma")
        param = {"gamma": [0, 0.5, 1, 1.5, 2, 2.5]}
        clf_gscv = GridSearchCV(estimator=clf, param_grid=param, scoring=scoring, n_jobs=nthread, iid=False, cv=nfold)
        
        t0 = pd.Timestamp.now()
        model_6 = clf_gscv.fit(x_train, y_train)
        t1 = pd.Timestamp.now()
        best_param = model_6.best_params_["gamma"]
        print("gamma：%s，用时：%s" % (best_param, (t1-t0)))
        time.sleep(60)
        
        # 7-精确搜索：gamma
        print("7-精确搜索：gamma")
        param = {"gamma": np.arange(best_param, best_param+1, 0.1)}
        clf_gscv = GridSearchCV(estimator=clf, param_grid=param, scoring=scoring, n_jobs=nthread, iid=False, cv=nfold)
        
        t0 = pd.Timestamp.now()
        model_7 = clf_gscv.fit(x_train, y_train)
        t1 = pd.Timestamp.now()
        best_param = model_7.best_params_["gamma"]
        clf.set_params(gamma=best_param)
        xgb_param = clf.get_xgb_params()
        print("gamma：%s，用时：%s" % (best_param, (t1-t0)))
        time.sleep(60)
        
        # 8-调整最优弱分类器个数：n_estimators_3
        print("8-调整最优弱分类器个数：n_estimators_3")
        clf.set_params(n_estimators=n_estimators_0)
        xgb_param = clf.get_xgb_params()
        
        t0 = pd.Timestamp.now()
        xgb_cv = xgb.cv(params=xgb_param, 
                        dtrain=d_train, 
                        num_boost_round=xgb_param["n_estimators"],
                        nfold=nfold, 
                        metrics=eval_metric, 
                        early_stopping_rounds=int(xgb_param["n_estimators"]/10),
                        verbose_eval=None)
        t1 = pd.Timestamp.now()
        n_estimators_3 = xgb_cv.shape[0]
        clf.set_params(n_estimators=n_estimators_3)
        xgb_param = clf.get_xgb_params()
        print("分类器个数：%s， 用时：%s" % (n_estimators_3, (t1-t0)))
        time.sleep(60)
        
        # 9-暴力搜索：subsample, colsample_bytree
        print("9-暴力搜索：subsample, colsample_bytree")
        param = {"subsample": [0.7,0.8,0.9], "colsample_bytree": [0.7,0.8,0.9]}
        clf_gscv = GridSearchCV(estimator=clf, param_grid=param, scoring=scoring, n_jobs=nthread, iid=False, cv=nfold)
        
        t0 = pd.Timestamp.now()
        model_8 = clf_gscv.fit(x_train, y_train)
        t1 = pd.Timestamp.now()
        best_param_1 = model_8.best_params_["subsample"]
        best_param_2 = model_8.best_params_["colsample_bytree"]
        print("subsample：%s，colsample_bytree：%s，用时：%s" % (best_param_1, best_param_2, (t1-t0)))
        time.sleep(60)
        
        # 10-精确搜索：subsample, colsample_bytree
        print("10-精确搜索：subsample, colsample_bytree")
        param = {"subsample": [best_param_1-0.05, best_param_1, best_param_1+0.05],
                 "colsample_bytree": [best_param_2-0.05, best_param_2, best_param_2+0.05]}
        clf_gscv = GridSearchCV(estimator=clf, param_grid=param, scoring=scoring, n_jobs=nthread, iid=False, cv=nfold)
        
        t0 = pd.Timestamp.now()
        model_9 = clf_gscv.fit(x_train, y_train)
        t1 = pd.Timestamp.now()
        best_param_1 = model_9.best_params_["subsample"]
        best_param_2 = model_9.best_params_["colsample_bytree"]
        clf.set_params(subsample=best_param_1, colsample_bytree=best_param_2)
        xgb_param = clf.get_xgb_params()
        print("subsample：%s，colsample_bytree：%s，用时：%s" % (best_param_1, best_param_2, (t1-t0)))
        time.sleep(60)
        
        # 11-精确搜索：reg_alpha
        print("11-精确搜索：reg_alpha")
        param = {"reg_alpha": np.arange(0, 1, 0.1)}
        clf_gscv = GridSearchCV(estimator=clf, param_grid=param, scoring=scoring, n_jobs=nthread, iid=False, cv=nfold)
        
        t0 = pd.Timestamp.now()
        model_11 = clf_gscv.fit(x_train, y_train)
        t1 = pd.Timestamp.now()
        best_param = model_11.best_params_["reg_alpha"]
        clf.set_params(reg_alpha=best_param)
        xgb_param = clf.get_xgb_params()
        print("reg_alpha：%s，用时：%s" % (best_param, (t1-t0)))
        time.sleep(60)
        
        # 12-精确搜索：max_delta_step, scale_pos_weight
        print("12-精确搜索：max_delta_step, scale_pos_weight")
        param = {"max_delta_step": np.arange(0, 0.1, 0.02), 
                 "scale_pos_weight": np.arange(1, 2, 0.2)}
        clf_gscv = GridSearchCV(estimator=clf, param_grid=param, scoring=scoring, n_jobs=nthread, iid=False, cv=nfold)
        
        t0 = pd.Timestamp.now()
        model_12 = clf_gscv.fit(x_train, y_train)
        t1 = pd.Timestamp.now()
        best_param_1 = model_12.best_params_["max_delta_step"]
        best_param_2 = model_12.best_params_["scale_pos_weight"]
        clf.set_params(max_delta_step=best_param_1, scale_pos_weight=best_param_2)
        xgb_param = clf.get_xgb_params()
        print("max_delta_step：%s，scale_pos_weight：%s，用时：%s" % (best_param_1, best_param_2, (t1-t0)))
        time.sleep(60)
        
        # 13-调整最优弱分类器个数：n_estimators_4
        print("13-调整最优弱分类器个数：n_estimators_4")
        clf.set_params(n_estimators=n_estimators_0)
        xgb_param = clf.get_xgb_params()
        
        t0 = pd.Timestamp.now()
        xgb_cv = xgb.cv(params=xgb_param, 
                        dtrain=d_train, 
                        num_boost_round=xgb_param["n_estimators"],
                        nfold=nfold, 
                        metrics=eval_metric, 
                        early_stopping_rounds=int(xgb_param["n_estimators"]/10),
                        verbose_eval=None)
        t1 = pd.Timestamp.now()
        n_estimators_4 = xgb_cv.shape[0]
        clf.set_params(n_estimators=n_estimators_4)
        xgb_param = clf.get_xgb_params()
        print("分类器个数：%s， 用时：%s" % (n_estimators_4, (t1-t0)))
        time.sleep(30)
        
        # 14-精确搜索：learning_rate
        print("14-精确搜索：learning_rate")
        lr = xgb_param["learning_rate"]
        param = {"learning_rate": np.arange(lr-0.1, lr+0.1, 0.05)}
        clf_gscv = GridSearchCV(estimator=clf, param_grid=param, scoring=scoring, n_jobs=nthread, iid=False, cv=nfold)
        
        t0 = pd.Timestamp.now()
        model_14 = clf_gscv.fit(x_train, y_train)
        t_1 = pd.Timestamp.now()
        best_param = model_14.best_params_["learning_rate"]
        clf.set_params(learning_rate=best_param)
        xgb_param = clf.get_xgb_params()
        print("learning_rate：%s，用时：%s" % (best_param, (t_1-t0)))
        time.sleep(60)
        
        # 15-终极训练
        print("15-终极训练")
        model_res = xgb.train(params=xgb_param,
                              dtrain=d_train,
                              num_boost_round=xgb_param["n_estimators"],
                              evals=watchlist,
                              early_stopping_rounds=int(xgb_param["n_estimators"]/10))
        t_end = pd.Timestamp.now()
        print("参数训练完毕，总用时：%s" % (t_end-t_begin))
        return model_res, clf

    def test(self, x_test, y_test, model_res, proba):
        d_test = xgb.DMatrix(x_test, y_test)
        y_hat = model_res.predict(d_test)
        y_pred = np.where(y_hat >= proba, 1, 0)
        
        t = pd.crosstab(y_test, y_pred)
        TP = t.iloc[0,0]
        #FP = t.iloc[1,0]
        FN = t.iloc[0,1]
        #TN = t.iloc[1,1]
        #precise = TP / (TP+FP)
        recall = TP / (TP+FN)
        #F1 = (2*precise*recall) / (precise+recall)
        print(t)
        print("召回率：%s" % str(recall))
        #print("精确率：%d" % precise)
        #print("F1：%d" % F1)
        return t, recall
    
    def predict(self):
        pass
    
    def data_pmml(self, df_train, clf):
        # 1-划分数据集
        x = df_train.iloc[:,1:]
        y = df_train.iloc[:,0]
        # 2-特征变换
        mapper = DataFrameMapper(
            features=[
                    (["contractprice"], None),
                    (["pre_hours"], None),
                    (["room_nights"], None),
                    (["orderattribute"], OneHotEncoder()),
                    (["ordersource"], OneHotEncoder()),
                    (["payway"], OneHotEncoder()),
                    (["cityid"], LabelBinarizer()),
                    (["hotelstargrade"], OneHotEncoder()),
                    (["contractprice_disperse"], LabelBinarizer()),
                    (["guaranstatus"], OneHotEncoder()),
                    (["create_late"], OneHotEncoder()),
                    (["create_sameday"], OneHotEncoder()),
                    (["create_twoday"], OneHotEncoder()),
                    (["pre_hours_disperse"], LabelBinarizer()),
                    (["room_nights_disperse"], LabelBinarizer()),
                    (["create_month"], LabelBinarizer()),
                    (["create_day"], OneHotEncoder()),
                    (["create_hour"], LabelBinarizer()),
                    (["create_weekday"], LabelBinarizer()),
                    (["checkin_month"], LabelBinarizer()),
                    (["checkin_day"], OneHotEncoder()),
                    (["checkin_weekday"], LabelBinarizer())
                    ],
            default=False
            )
        # 3-PMMLPipeline
        pipeline = PMMLPipeline([
            ("columns", mapper),
            ("XGBClassifier", clf)
            ])
        pipeline.fit(x, y)
        return pipeline
