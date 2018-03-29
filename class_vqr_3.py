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
                    (["orderattribute"], None),
                    (["ordersource"], OneHotEncoder()),
                    (["payway"], OneHotEncoder()),
                    (["cityid"], LabelBinarizer()),
                    (["hotelstargrade"], OneHotEncoder()),
                    (["contractprice"], None),
                    (["contractprice_disperse"], LabelBinarizer()),
                    (["guaranstatus"], None),
                    (["create_late"], None),
                    (["create_sameday"], None),
                    (["create_twoday"], None),
                    (["pre_hours"], None),
                    (["pre_hours_disperse"], LabelBinarizer()),
                    (["room_nights"], None),
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
        t1 = pd.Timestamp.now()
        print("特征工程处理完毕，用时：%s" % (t1-t0))
        return df_train, df_train_transform, df_test_transform, pre_hours, room_nights, price, mapper_fit
