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
