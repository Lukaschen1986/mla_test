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
      
      def data_pmml(self, df_train, clf):
        # 1-划分数据集
        x = df_train.iloc[:,1:]
        y = df_train.iloc[:,0]
        # 2-特征变换
        mapper = DataFrameMapper(
            features=[
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
        # 3-PMMLPipeline
        pipeline = PMMLPipeline([
            ("columns", mapper),
            ("XGBClassifier", clf)
            ])
        pipeline.fit(x, y)
        return pipeline
