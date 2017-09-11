# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import pandas as pd
import random as rd
#import matplotlib.pyplot as plt
#from scipy.spatial import distance
from scipy import stats
from sklearn.cluster import KMeans
from sklearn import preprocessing

def runif(n, xmin, xmax):
    res = []
    for i in range(n):
        val = rd.uniform(xmin, xmax)
        res.append(val)
    res = np.array(res)
    return res

def GMM_Kmeans(df_regular, k=3, steps=100, label_prob=0.9):
    #df_regular = air_regular
    df_sca = pd.DataFrame(preprocessing.scale(df_regular, axis=0))
    nrow = len(df_sca); ncol = len(df_sca.columns)
    kmeans = KMeans(n_clusters=k, max_iter=5000, random_state=123).fit(df_sca) # random_state
    # km_mse
    km_mse = kmeans.inertia_/nrow
    #df_mse = df_sca
    #df_mse["km_label"] = kmeans.labels_
    #km_centers = pd.DataFrame(kmeans.cluster_centers_)
    #mse = 0.0
    #for i in xrange(k):
        #df_mse_tmp = df_mse[df_mse.km_label == i]
        #mse += np.sum((df_mse_tmp.iloc[:,0:len(df_mse_tmp.columns)-1]-km_centers.iloc[i,:])**2).sum()
    #km_mse = mse/nrow
    
    mu_group = pd.DataFrame(kmeans.cluster_centers_)
    mu_group.columns = df_sca.columns
    
    sigma_group = df_sca.apply(np.std, axis=0)
    sigma_group = pd.DataFrame(np.tile(sigma_group,k).reshape(k,ncol))
    sigma_group.columns = df_sca.columns
    
    P = runif(n=k, xmin=0.0, xmax=1.0)
    pdf = np.ones((nrow,k))
    Q = np.zeros((nrow,k))
    
    mu_new = pd.DataFrame(np.zeros((k,ncol)))
    sigma_new = pd.DataFrame(np.zeros((k,ncol)))
    P_new = np.tile(0.0, k)
    e = 10**-5
    
    for step in range(steps):
        # E
        for j in range(k):
            for col in df_sca.columns:
                pdf[:,j] *= stats.norm.pdf(df_sca[col], mu_group[col][j], sigma_group[col][j])
        for row in range(len(pdf)):
            if sum(pdf[row,:]) == 0.0:
                pdf[row,:] = 0.001
        for i in range(nrow): Q[i,:] = pdf[i,:]*P/np.dot(pdf[i,:],P)
        # M
        for j in range(k):
            mu_new.iloc[j,:] = np.dot(Q[:,j],df_sca)/np.sum(Q[:,j])
            sigma_new.iloc[j,:] = np.sqrt(np.dot(Q[:,j],(df_sca-mu_group.iloc[j,:])**2)/np.sum(Q[:,j]))
            P_new[j] = np.sum(Q[:,j])/nrow
            mu_new.columns = df_sca.columns
            sigma_new.columns = df_sca.columns
        # epsilon
        e_mu = np.array(np.sum(np.abs(mu_new-mu_group))).sum()/len(df_sca.columns)
        e_sigma = np.array(np.sum(np.abs(sigma_new-sigma_group))).sum()/len(df_sca.columns)
        e_P = np.sum(np.abs(P_new-P))
        if e_mu <= e and e_sigma <= e and e_P <= e:
            break
        else:
            mu_group = mu_new
            sigma_group = sigma_new
            P = P_new
    
    df_Q = pd.DataFrame(Q)
    df_Q["row_max"] = df_Q.apply(max, axis=1)
    label = []
    for i in range(len(df_Q)):
        if df_Q.row_max[i] >= label_prob:
            val = df_Q.columns[np.where(df_Q.row_max[i] == df_Q.iloc[i,0:len(df_Q.columns)-1])[0][0]]
            val = str(val)
        else:
            val = "other"
        label.append(val)
    # gmm_mse
    df_mse = df_sca
    df_mse["gmm_label"] = label
    df_mse = df_mse[df_mse.gmm_label != "other"]
    df_mse.gmm_label = df_mse.gmm_label.astype(int)
    gmm_centers = df_mse.groupby("gmm_label",as_index=False).mean()
    gmm_centers = gmm_centers.drop("gmm_label",axis=1)
    mse = 0.0
    for i in range(k):
        df_mse_tmp = df_mse[df_mse.gmm_label == i]
        mse += np.sum((df_mse_tmp.iloc[:,0:len(df_mse_tmp.columns)-1]-gmm_centers.iloc[i,:])**2).sum()
    gmm_mse = mse/nrow
    return label, km_mse, gmm_mse
