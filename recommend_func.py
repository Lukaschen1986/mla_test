# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import math
import pymysql
import gc
from datetime import *
from scipy.spatial import distance

# 把列名变为小写函数
def lower_colname(df):
    colname = []
    for col in df.columns:
        lower_col = str(col).lower()
        colname.append(lower_col)
    df.columns = colname
    return df

def dataPreProcess(conn, statis_date, secretary_sql, hotel_sql):
    secretary_info = pd.read_sql(secretary_sql, conn)
    secretary_info = lower_colname(secretary_info)
    hotel_order = pd.read_sql(hotel_sql, conn)
    hotel_order = lower_colname(hotel_order)
    dataSet = hotel_order[(hotel_order["clientremark"] != u"续住") & \
                          (-hotel_order["memberid"].isin(secretary_info["memberid"]))]
    dataSet.longitude = np.round(dataSet.longitude.astype(float),3) # 格式化经度
    dataSet.latitude = np.round(dataSet.latitude.astype(float),3) # 格式化纬度
    return dataSet

def cityList(dataSet):
    city_order = pd.DataFrame(dataSet["cityid"].value_counts()) # 所有城市列表
    city_order.columns = ["order_sum"]
    city_order = city_order[city_order.index != u""]
    city_hotel = dataSet[["cityid","hotelid"]].drop_duplicates()
    city_hotel = city_hotel.groupby("cityid",as_index=False)[["hotelid"]].count()
    city_hotel = city_hotel[city_hotel.cityid != u""]
    city_hotel.columns = city_hotel.columns.str.replace("hotelid","hotel_sum")
    city_list = pd.merge(city_order, city_hotel, left_index=True, right_on="cityid", how="inner")
    city_list = city_list.set_index("cityid")
    city_list_big = city_list[(city_list.order_sum >= 30) & (city_list.hotel_sum >= 10)]
    return city_list, city_list_big

def hotFav(city_list, dataSet, k=10):
    hot_fav_final = pd.DataFrame()
    for idx in xrange(len(city_list)):
        # idx = 0
        df_sub = dataSet[dataSet.cityid == city_list.index[idx]]
        # 提取数据子集中每个酒店的星级和经纬度
        hotel_list = df_sub[["hotelid","hotelstargrade","longitude","latitude"]].drop_duplicates()
        hotel_list = hotel_list.sort_values(by=["hotelid","hotelstargrade"], ascending=False)
        key = [True]
        for i in xrange(1,len(hotel_list)):
            # i = 1
            if hotel_list.hotelid.iloc[i] == hotel_list.hotelid.iloc[i-1]:
                val = False
            else:
                val = True
            key.append(val)
        hotel_list = hotel_list[key]
        hotel_list.columns = hotel_list.columns.str.replace("hotelid","itemid")
        hotel_list.hotelstargrade[hotel_list.hotelstargrade == u""] = np.nan
        hotel_list.longitude[hotel_list.longitude == 0] = np.nan
        hotel_list.latitude[hotel_list.latitude == 0] = np.nan
        # hot_hotel
        hot_hotel = df_sub.groupby("hotelid", as_index=False)[["orderid"]].count()
        hot_hotel = hot_hotel.sort_values(by="orderid", ascending=False).head(k)
        hot_hotel = hot_hotel.dropna(how="any", axis=0)
        hot_hotel = pd.DataFrame({"cityid": city_list.index[idx],
                                  "itemid": hot_hotel.hotelid,
                                  "val": hot_hotel.orderid})
        hot_hotel = pd.merge(hot_hotel, hotel_list, on="itemid", how="left")
        hot_hotel.to_csv("hot_hotel.csv", header=False, mode="a+")
        hot_fav_final = pd.concat((hot_fav_final, hot_hotel), axis=0, ignore_index=True)
    return hot_fav_final

def yourFav(city_list, dataSet, statis_date, alpha=0.001):
    your_fav_final = pd.DataFrame()
    for idx in xrange(len(city_list)):
        # idx = 30
        df_sub = dataSet[dataSet.cityid == city_list.index[idx]]
        # 提取数据子集中每个酒店的星级和经纬度
        hotel_list = df_sub[["hotelid","hotelstargrade","longitude","latitude"]].drop_duplicates()
        hotel_list = hotel_list.sort_values(by=["hotelid","hotelstargrade"], ascending=False)
        key = [True]
        for i in xrange(1,len(hotel_list)):
            # i = 1
            if hotel_list.hotelid.iloc[i] == hotel_list.hotelid.iloc[i-1]:
                val = False
            else:
                val = True
            key.append(val)
        hotel_list = hotel_list[key]
        hotel_list.columns = hotel_list.columns.str.replace("hotelid","itemid")
        hotel_list.hotelstargrade[hotel_list.hotelstargrade == u""] = np.nan
        hotel_list.longitude[hotel_list.longitude == 0] = np.nan
        hotel_list.latitude[hotel_list.latitude == 0] = np.nan
        # your_fav
        your_fav_order = df_sub.groupby(["memberid","hotelid"], as_index=False)[["orderid"]].count()
        your_fav_checkout = df_sub.groupby(["memberid","hotelid"], as_index=False)[["recheckoutdate"]].max()
        your_fav = pd.merge(your_fav_order, your_fav_checkout, on=["memberid","hotelid"], how="inner")
        time_pass = []
        for date in your_fav["recheckoutdate"]:
            x = (statis_date-pd.Timestamp(date)).days
            time_pass.append(x)
        your_fav["time_pass"] = time_pass
        your_fav["val"] = your_fav["orderid"]*np.exp(-alpha*your_fav["time_pass"])
        your_fav.columns = your_fav.columns.str.replace("hotelid","itemid")
        your_fav = pd.merge(your_fav, hotel_list, on="itemid", how="left")
        your_fav = pd.DataFrame({"cityid": city_list.index[idx],
                                 "memberid": your_fav.memberid,
                                 "itemid": your_fav.itemid,
                                 "val": your_fav.val,
                                 "hotelstargrade": your_fav.hotelstargrade,
                                 "longitude": your_fav.longitude,
                                 "latitude": your_fav.latitude},
        columns=["cityid","memberid","itemid","val","hotelstargrade","longitude","latitude"])
        your_fav = your_fav.sort_values(by=["memberid","val"], ascending=False)
        your_fav.to_csv("your_fav.csv", header=False, mode="a+")
        your_fav_final = pd.concat((your_fav_final, your_fav), axis=0, ignore_index=True)
    return your_fav_final
#your_fav_final = your_fav

def guessFav(city_list_big, your_fav_final, K=10, steps=100, a=0.01, b=0.001):
    guess_fav_final = pd.DataFrame()
    for idx in xrange(len(city_list_big)):
        # idx = 30
        df_sub = your_fav_final[your_fav_final.cityid == city_list_big.index[idx]]
#        df_sub = your_fav_final
        # 提取数据子集中每个酒店的星级和经纬度
        hotel_list = df_sub[["itemid","hotelstargrade","longitude","latitude"]].drop_duplicates()
        hotel_list = hotel_list.sort_values(by=["itemid","hotelstargrade"], ascending=False)
        key = [True]
        for i in xrange(1,len(hotel_list)):
            # i = 1
            if hotel_list.itemid.iloc[i] == hotel_list.itemid.iloc[i-1]:
                val = False
            else:
                val = True
            key.append(val)
        hotel_list = hotel_list[key]
        hotel_list.columns = hotel_list.columns.str.replace("itemid","itemid")
        hotel_list.hotelstargrade[hotel_list.hotelstargrade == u""] = np.nan
        hotel_list.longitude[hotel_list.longitude == 0] = np.nan
        hotel_list.latitude[hotel_list.latitude == 0] = np.nan
        # LFM
        R = pd.pivot_table(df_sub, index="memberid", columns="itemid", values="val", fill_value=0, aggfunc=np.sum)
        R_hat, E_in, E_in_append = LFM_model(R, userID=R.index, itemID=R.columns, K, steps, a, b)
#        print E_in, plt.plot(E_in_append)
        ## ICF_matrix
        sim_corr = corr_matrix(df=R, yita=10) # 计算相关系数矩阵
        dist_col = -np.log(sim_corr/2 + 0.5) # 转化为距离
        dist_col = pd.DataFrame(dist_col, index=R.columns, columns=R.columns)
        ## UCF_matrix
        dist_row = distance.cdist(R, R, metric='cosine') # 计算余弦距离矩阵
        dist_row = pd.DataFrame(dist_row, index=R.index, columns=R.index)
        gc.collect()
        ## 计算模型
        mem_list = np.unique(df_sub["memberid"])
        list_recommend_final = pd.DataFrame()
        for m in mem_list:
            # m = mem_list[0]
            icf_list = ICF_model(df = df_sub, memberid = m, dist_df = dist_col, k = 10) # icf模型
            ucf_list = UCF_model(df = df_sub, memberid = m, dist_df = dist_row, k = 10) # ucf模型
            if icf_list is None: icf_list = pd.DataFrame({"itemid":[], "icf_score":[]})
            if ucf_list is None: ucf_list = pd.DataFrame({"itemid":[], "ucf_score":[]})    
            lfm_list = LFM_model_list(df = df_sub, R_hat = R_hat, memberid = m)
            res_list = pd.merge(icf_list, ucf_list, on="itemid", how="outer")
            res_list = pd.merge(res_list, lfm_list, on="itemid", how="outer")
            res_list = res_list.dropna(how="all", axis=0) # 删除全部为缺失值的行
            res_list = res_list.fillna(0)
            res_list_scale = res_list.drop("itemid", axis=1).apply(lambda x: (x-min(x))/(max(x)-min(x)), axis=0)
            res_list_scale["score"] = res_list_scale.apply(np.mean, axis=1)
            res_list = pd.concat((res_list.itemid, res_list_scale), axis=1)
            res_list = res_list.sort_values(by="score",ascending=False)
            if len(res_list["itemid"]) == 0:
                itemid = []
            else:
                itemid = res_list["itemid"]
            if len(res_list["score"]) == 0:
                score = []
            else:
                score = res_list["score"]
            list_recommend = pd.DataFrame({"cityid":city_list_big.index[idx],
                                           "memberid":m,
                                           "itemid":itemid,
                                           "score":score},columns=["cityid","memberid","itemid","score"])
            list_recommend = pd.merge(list_recommend, hotel_list, on="itemid", how="left")
            list_recommend = list_recommend.sort_values(by=["memberid","score"], ascending=False)
            list_recommend.to_csv("list_recommend.csv", header=False, mode="a+")
            list_recommend_final = pd.concat((list_recommend_final, list_recommend), axis=0, ignore_index=True)
        guess_fav_final = pd.concat((guess_fav_final, list_recommend_final), axis=0, ignore_index=True)
    return guess_fav_final
#guess_fav_final = list_recommend_final

# 城市热门酒店函数
#df=df_sub; city_list=city_list; hotel_list=hotel_list; k=10; idx=idx
#def hotFav(df, city_list, hotel_list, k, idx):
#    hot_hotel = df.groupby("hotelid", as_index=False)[["orderid"]].count()
#    hot_hotel = hot_hotel.sort_values(by="orderid", ascending=False).head(k)
#    hot_hotel = hot_hotel.dropna(how="any", axis=0)
#    hot_hotel = pd.DataFrame({"cityid": city_list.index[idx],
#                              "itemid": hot_hotel.hotelid,
#                              "val": hot_hotel.orderid})
#    hot_hotel = pd.merge(hot_hotel, hotel_list, on="itemid", how="left")
#    return hot_hotel
#
##df=df_sub; city_list=city_list; hotel_list=hotel_list; statis_date=pd.Timestamp("2017-03-04"); idx=idx; alpha=0.001
#def yourFav(df, city_list, hotel_list, statis_date, idx, alpha):
#    your_fav_order = df.groupby(["memberid","hotelid"], as_index=False)[["orderid"]].count()
#    your_fav_checkout = df.groupby(["memberid","hotelid"], as_index=False)[["recheckoutdate"]].max()
#    your_fav = pd.merge(your_fav_order, your_fav_checkout, on=["memberid","hotelid"], how="inner")
#    time_pass = []
#    for date in your_fav["recheckoutdate"]:
#        x = (statis_date-pd.Timestamp(date)).days
#        time_pass.append(x)
#    your_fav["time_pass"] = time_pass
#    your_fav["val"] = your_fav["orderid"]*np.exp(-alpha*your_fav["time_pass"])
#    your_fav.columns = your_fav.columns.str.replace("hotelid","itemid")
#    your_fav = pd.merge(your_fav, hotel_list, on="itemid", how="left")
#    your_fav = pd.DataFrame({"cityid": city_list.index[idx],
#                             "memberid": your_fav.memberid,
#                             "itemid": your_fav.itemid,
#                             "val": your_fav.val,
#                             "hotelstargrade": your_fav.hotelstargrade,
#                             "longitude": your_fav.longitude,
#                             "latitude": your_fav.latitude},
#    columns=["cityid","memberid","itemid","val","hotelstargrade","longitude","latitude"])
#    return your_fav

# 含惩罚因子的相关系数函数
def corr_parameter(x1, x2, yita):
    penalty = 1/np.log2(2+yita*sum((x1 != 0) & (x2 != 0)))
    corr = penalty*np.corrcoef(x1,x2)[0][1]
    return corr

# 相关系数矩阵
def corr_matrix(df, yita):
    mtrx = np.zeros((len(df.columns),len(df.columns)))
    for i in xrange(len(df.columns)):
        for j in xrange(len(df.columns)):
            mtrx[i,j] = corr_parameter(x1=df.iloc[:,i], x2=df.iloc[:,j], yita=yita)
    return mtrx

# ICF协同过滤模型
#df = your_fav; memberid = m; dist_df = dist_col; k = 10
def ICF_model(df, memberid, dist_df, k):
    item_tab = df[df["memberid"] == memberid] # 用户历史商品列表
    item_tab.index = range(len(item_tab))
    df_icf = pd.DataFrame()
    for i in xrange(len(item_tab)):
        # i = 0
        k_df = dist_df.loc[item_tab.itemid[i],:].sort_values(ascending=True).head(k+1) # 取距离最近的k个商品
        k_df = k_df.dropna(how="any",axis=0) # 剔除缺失值
        k_df = pd.DataFrame({"itemid": k_df.index, "dist": k_df.values},columns=["itemid","dist"]) # 合并距离数据
        k_df["sim"] = 1/k_df["dist"] # 将距离转化为相似性
        k_df["item_num"] = item_tab.val[i] # 插入变量信息
        k_df = k_df[-k_df.itemid.isin(item_tab.itemid)] # 删除已购历史商品
        df_icf = pd.concat([df_icf,k_df], axis=0)
    if len(df_icf) == 0: return None
    df_icf["icf_score"] = df_icf.sim*df_icf.item_num # 计算得分
    df_icf_groupby = df_icf.groupby("itemid", as_index=False)[["icf_score"]].sum() # 按推荐商品聚合
    df_icf_groupby = df_icf_groupby.sort_values(by="icf_score",ascending=False) # 倒序
    return df_icf_groupby

# UCF协同过滤模型
def UCF_model(df, memberid, dist_df, k):
    item_tab = df[df["memberid"] == memberid] # 用户历史商品列表
    item_tab.index = range(len(item_tab))
    k_df = dist_df.loc[memberid,:].sort_values(ascending=True).head(k+1) # 取距离最近的k个用户
    k_df = k_df.dropna(how="any",axis=0)
    k_df = pd.DataFrame({"recommend_user": k_df.index, "dist": k_df.values},columns=["recommend_user","dist"])
#    k_df = k_df.drop([0])
    k_df["sim"] = 1-k_df["dist"]
    recommend_user_list = pd.DataFrame()
    for i in range(len(k_df)): # 遍历每个相似用户，取相似用户的历史偏好商品，剔除已购历史商品
        # i = 0
        item_tab_recommend_user = df[df.memberid == k_df.recommend_user[i]]
        item_tab_recommend_user = pd.DataFrame({"recommend_user":k_df.recommend_user[i],
                                                "itemid": item_tab_recommend_user.itemid,
                                                "item_num": item_tab_recommend_user.val})
        item_tab_recommend_user = item_tab_recommend_user[-item_tab_recommend_user.itemid.isin(item_tab.itemid)]
        recommend_user_list = pd.concat((recommend_user_list,item_tab_recommend_user), axis=0)
    if len(recommend_user_list) == 0: return None
    df_ucf = pd.merge(k_df, recommend_user_list, on="recommend_user", how="inner")
    df_ucf["ucf_score"] = df_ucf["sim"]*df_ucf["item_num"]
    df_ucf_groupby = df_ucf.groupby("itemid", as_index=False)[["ucf_score"]].sum()
    df_ucf_groupby = df_ucf_groupby.sort_values(by="ucf_score",ascending=False)
    return(df_ucf_groupby)

# LFM隐语义模型
#def LFM_model(R, userID, itemID, K, steps, a, b):
#    # 初始化参数q,p矩阵, 随机
#    arrayp = np.random.rand(len(userID), K)
#    arrayq = np.random.rand(K, len(itemID))
#    P = pd.DataFrame(arrayp, columns=range(0,K), index=userID)
#    Q = pd.DataFrame(arrayq, columns=itemID, index=range(0,K))
#    # LFM
#    E_in_append = []
#    for step in xrange(steps):
#        for i in xrange(len(R)):
#            for j in xrange(len(R.columns)):
#                # i = 0; j = 3
#                if R.iloc[i,j] > 0:
#                    eij = R.iloc[i,j] - np.dot(P.iloc[i,:], Q.iloc[:,j])
#                    P.iloc[i,:] += a*(2*eij*Q.iloc[:,j]-b*P.iloc[i,:])
#                    Q.iloc[:,j] += a*(2*eij*P.iloc[i,:]-b*Q.iloc[:,j])
#        R_hat = pd.DataFrame(np.dot(P, Q), index=userID, columns=itemID)
#        eij_hat = 0; E_in = 0
#        for i in xrange(len(R)):
#            for j in xrange(len(R.columns)):
#                if R.iloc[i,j] > 0:
#                    eij_hat = R.iloc[i,j] - R_hat.iloc[i,j]
#                    E_in += eij_hat**2 + (b/2)*P.iloc[i,:].dot(P.iloc[i,:]) + (b/2)*Q.iloc[:,j].dot(Q.iloc[:,j])
#        E_in_append.append(E_in)
#        if round(E_in,1) <= 0.8:
#            break
#    return R_hat, E_in, E_in_append

def LFM_model(R, userID, itemID, K, steps, a, b):
    # 初始化参数q,p矩阵, 随机
    arrayp = np.random.rand(len(userID), K)
    arrayq = np.random.rand(K, len(itemID))
    P = pd.DataFrame(arrayp, columns=range(0,K), index=userID)
    Q = pd.DataFrame(arrayq, columns=itemID, index=range(0,K))
    # LFM
    E_in_append = []
    for step in xrange(steps):
        eij_hat = 0; E_in = 0
        for i in xrange(len(R)):
            for j in xrange(len(R.columns)):
                # i = 0; j = 1
                if R.iloc[i,j] > 0:
                    # eij
                    eij = R.iloc[i,j] - np.dot(P.iloc[i,:], Q.iloc[:,j])
                    P.iloc[i,:] += a*(2*eij*Q.iloc[:,j]-b*P.iloc[i,:])
                    Q.iloc[:,j] += a*(2*eij*P.iloc[i,:]-b*Q.iloc[:,j])
                    # eij_hat
                    eij_hat = R.iloc[i,j] - np.dot(P.iloc[i,:], Q.iloc[:,j])
                    E_in += eij_hat**2 + (b/2)*P.iloc[i,:].dot(P.iloc[i,:]) + (b/2)*Q.iloc[:,j].dot(Q.iloc[:,j])
        E_in_append.append(E_in)
        R_hat = pd.DataFrame(np.dot(P, Q), index=userID, columns=itemID)
        if round(E_in,1) <= 1: break
    return R_hat, E_in, E_in_append
#R_hat, E_in, E_in_append = LFM_model(R=df_pivot, userID=df_pivot.index, itemID=df_pivot.columns, K=10, steps=200, a=0.01, b=0.001)

# LFM隐语义模型2：筛选item名单
def LFM_model_list(df, R_hat, memberid):
    item_tab = df[df["memberid"] == memberid]
    guess_list = pd.DataFrame({"itemid":R_hat.loc[memberid,:].sort_values(ascending=False).index,
                               "lfm_score":R_hat.loc[memberid,:].sort_values(ascending=False)},
    columns=["itemid","lfm_score"])
    guess_list = guess_list[-guess_list.itemid.isin(item_tab.itemid)]
    guess_list = guess_list.head(10)
    guess_list = guess_list.dropna(how="any",axis=0) # 剔除缺失值
    guess_list.index = range(len(guess_list))
    return guess_list

#df=your_fav; R=R; R_hat=R_hat; yita=10
#def guessFav(df, R, R_hat, yita):
#    ## ICF_matrix
#    sim_corr = corr_matrix(df=R, yita=yita) # 计算相关系数矩阵
#    dist_col = -np.log(sim_corr/2 + 0.5) # 转化为距离
#    dist_col = pd.DataFrame(dist_col, index=R.columns, columns=R.columns)
#    ## UCF_matrix
#    dist_row = distance.cdist(R, R, metric='cosine') # 计算余弦距离矩阵
#    dist_row = pd.DataFrame(dist_row, index=R.index, columns=R.index)
#    gc.collect()
#    ## 计算模型
#    mem_list = np.unique(df["memberid"])
#    list_recommend_final = pd.DataFrame()
#    for m in mem_list:
#        # m = mem_list[2]
#        icf_list = ICF_model(df = your_fav, memberid = m, dist_df = dist_col, k = 10) # icf模型
#        ucf_list = UCF_model(df = your_fav, memberid = m, dist_df = dist_row, k = 10) # ucf模型
#        if icf_list is None: icf_list = pd.DataFrame({"itemid":[], "icf_score":[]})
#        if ucf_list is None: ucf_list = pd.DataFrame({"itemid":[], "ucf_score":[]})    
#        lfm_list = LFM_model_list(df = your_fav, R_hat = R_hat, memberid = m)
#        res_list = pd.merge(icf_list, ucf_list, on="itemid", how="outer")
#        res_list = pd.merge(res_list, lfm_list, on="itemid", how="outer")
#        res_list = res_list.dropna(how="all", axis=0) # 删除全部为缺失值的行
#        res_list = res_list.fillna(0)
#        res_list_scale = res_list.drop("itemid", axis=1).apply(lambda x: (x-min(x))/(max(x)-min(x)), axis=0)
#        res_list_scale["score"] = res_list_scale.apply(np.mean, axis=1)
#        res_list = pd.concat((res_list.itemid, res_list_scale), axis=1)
#        res_list = res_list.sort_values(by="score",ascending=False)
#        if len(res_list["itemid"]) == 0:
#            itemid = []
#        else:
#            itemid = res_list["itemid"]
#        if len(res_list["score"]) == 0:
#            score = []
#        else:
#            score = res_list["score"]
#        list_recommend = pd.DataFrame({"cityid":city_list.index[idx],
#                                       "memberid":m,
#                                       "itemid":itemid,
#                                       "score":score},columns=["cityid","memberid","itemid","score"])
#        list_recommend = pd.merge(list_recommend, hotel_list, on="itemid", how="left")
#        list_recommend_final = pd.concat((list_recommend_final, list_recommend), axis=0, ignore_index=True)
#    return list_recommend_final
