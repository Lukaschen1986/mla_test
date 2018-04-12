n_estimators_0 = 200
clf = XGBClassifier(
        # General Parameters
        booster="gbtree", # gbtree: 基于树的模型; gblinear: 线性模型
        silent=0, # 0打印信息; 1不打印执行信息
        nthread=4, # 设置在几个核上并发
        n_jobs=4,
        # Booster Parameters
        learning_rate=0.3, # default=0.3
        n_estimators=n_estimators_0, # Number of boosted trees to fit
        gamma=0, # default=0, 只有当分裂使loss减小的值大于gamma，节点才分裂。设置为0，表示只要loss函数减少，就分裂
        max_depth=6, # default=6, 设置树的最大深度, 如果树的深度太大会导致过拟合, 应该使用CV来调节。Typical values: 3-10
        min_child_weight=1, # default=1, 这个参数用来控制过拟合, 应该使用CV来调节。
        subsample=0.8, # default=1, 每棵树随机采样的比例, 减小参数可防止过拟合, Typical values: 0.5-1
        colsample_bytree=0.8, # default=1, 每棵树列采样的比例, Typical values: 0.5-1
        reg_alpha=0, # default=0, L1正则化项, 可以应用在很高维度的情况下，使得算法的速度更快
        reg_lambda=1, # default=1, L2正则化项, 虽然大部分数据科学家很少用到这个参数
        max_delta_step=0, # default=0, 限制每棵树权重改变的最大步长, 如果参数设置为0，表示没有限制。如果设置为一个正值，会使得更新步更加谨慎和保守, 通常，这个参数不需要设置。但是当在逻辑回归中，各类别的样本十分不平衡时它是很有帮助的
        scale_pos_weight=1, # default=1, 在各类别样本十分不平衡时，把这个参数设定为一个正值，可以使算法更快收敛
        # Learning Task Parameters
        objective="binary:logistic", # binary:logistic; multi:softmax; multi:softprob
        eval_metric="auc", #  default according to objective: rmse, mae, logloss, error, merror, mlogloss, auc
#        num_class=len(set(y_train)),
        seed=1
        )
        
# 训练组合特征（备选）
xgb_param = clf.get_xgb_params()
d_train = xgb.DMatrix(x_train, y_train)
d_test = xgb.DMatrix(x_test, y_test)
watchlist = [(d_train, "train"), (d_test, "test")]
model_1 = xgb.train(params=xgb_param,
                    dtrain=d_train,
                    num_boost_round=n_estimators_1,
                    evals=watchlist,
                    early_stopping_rounds=None,
                    verbose_eval=True)
x_train_new = model_1.predict(d_train, pred_leaf=True)
x_test_new = model_1.predict(d_test, pred_leaf=True)

oht = OneHotEncoder().fit(x_train_new) # one-hot-encoding
x_train_new = oht.transform(x_train_new).toarray()
x_test_new = oht.transform(x_test_new).toarray()

x_train_new = pd.DataFrame(x_train_new, index=x_train.index)
x_test_new = pd.DataFrame(x_test_new, index=x_test.index)

x_train_update = pd.concat((x_train, x_train_new), axis=1).astype(np.float32) # 合并原始特征与组合特征
x_test_update = pd.concat((x_test, x_test_new), axis=1).astype(np.float32)

# 特征筛选（备选）
model_1 = xgb.train(params=xgb_param,
                    dtrain=d_train,
                    num_boost_round=n_estimators_1,
                    evals=watchlist,
                    early_stopping_rounds=None,
                    verbose_eval=True)

def create_feature_map(fmap_filename, features):
    outfile = open(fmap_filename, 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()
create_feature_map('D:/my_project/Python_Project/iTravel/virtual_quota_room/txt/model_res.fmap', feat_name)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5,5))
xgb.plot_importance(model_1, height=0.5, ax=ax)

fig, ax = plt.subplots(figsize=(30,30))
xgb.plot_tree(model_1, fmap="./model_res.fmap", num_trees=0, ax=ax)

fscore = model_1.get_fscore()
feat_imp = pd.DataFrame({"feat": list(fscore.keys()), "imp": list(fscore.values())})
feat_imp.imp /= np.sum(feat_imp.imp) # 归一化
feat_imp = feat_imp.sort_values(by="imp", ascending=False) # 特征降序
feat_imp["cumsum"] = np.cumsum(feat_imp.imp, axis=0)
feat_imp = feat_imp[feat_imp["cumsum"] < 0.99] # 特征筛选
feat_name = feat_imp.feat.tolist()
f = open("/home/bimining/project/itravel/virtual_quota_room/txt/feat_name.txt", "wb")
pickle.dump(feat_name, f); f.close()

x_train_filter = x_train.iloc[:, feat_imp.index]
x_test_filter = x_test.iloc[:, feat_imp.index]
