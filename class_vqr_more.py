def create_feature_map(fmap_filename, features):
    outfile = open(fmap_filename, 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()
create_feature_map('D:/my_project/Python_Project/iTravel/virtual_quota_room/txt/model_res.fmap', feat_name)

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

# 19-特征名称
df_train_transform_2 = pd.get_dummies(df_train, columns=["orderattribute","ordersource","payway","cityid","hotelstargrade",
                                                                 "contractprice_disperse","guaranstatus","create_late","create_sameday",
                                                                 "create_twoday","pre_hours_disperse","room_nights_disperse","create_month",
                                                                 "create_day","create_hour","create_weekday","checkin_month",
                                                                 "checkin_day","checkin_weekday"], drop_first=False, dummy_na=False)
feat_name = df_train_transform_2.columns[1:]
