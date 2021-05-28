# -*- coding: utf-8 -*-
"""
Created on Fri May 28 19:31:13 2021

@author: ASUS

卡方的意义：衡量理论与实际的差异程度
https://zhuanlan.zhihu.com/p/128905132
https://blog.csdn.net/ludan_xia/article/details/81737669
https://blog.csdn.net/qq_38214903/article/details/82967812
"""
import pandas as pd 
import numpy as np 
from scipy import stats

chi_squared_stats = lambda observed, expected: ((observed - expected)**2 / expected).sum().sum()

# 单组样本
data = pd.DataFrame({
    "observed": [23, 20, 18, 19, 24, 16],
    "expected": [20]*6
    }, index=["1点", "2点", "3点", "4点", "5点", "6点"])

# stats.chi2.cdf
statistic = chi_squared_stats(data.observed, data.expected)
df = (data.shape[0] - 1) * (data.shape[1] - 1)
bound = stats.chi2.ppf(q=0.95, df=df)
p_value = 1 - stats.chi2.cdf(x=statistic, df=df)
print(statistic)
print(bound)
print(p_value)

# stats.chisquare
statistic, p_value = stats.chisquare(f_obs=data.observed, f_exp=data.expected)
print(statistic)
print(p_value)


# 两组独立样本
observed = pd.DataFrame({
    "get_cold": [43, 28],
    "not_cold": [96, 84]
    }, index=["drink", "not"])

def get_expected(observed):
    observed_update = observed[:]
    observed_update["sum"] = observed_update.apply(np.sum, axis=1)
    observed_update = pd.concat([observed_update, 
                                 pd.DataFrame(observed_update.apply(np.sum, axis=0), columns=["sum"]).T])
    
    rate = observed_update.iloc[-1,0] / observed_update.iloc[-1,-1]
    expected = observed_update[:]
    expected.iloc[:,0] = observed_update.iloc[:,-1] * rate
    expected.iloc[:,1] = observed_update.iloc[:,-1] * (1 - rate)
    expected = expected.iloc[0:-1, 0:-1]
    return expected

# stats.chi2.cdf
expected = get_expected(observed)
statistic = chi_squared_stats(observed, expected)
df = (observed.shape[0] - 1) * (observed.shape[1] - 1)
bound = stats.chi2.ppf(q=0.95, df=df)
p_value = 1 - stats.chi2.cdf(x=statistic, df=df)
print(statistic)
print(bound)
print(p_value)

# stats.chi2_contingency
statistic, p_value, df, expected = stats.chi2_contingency(observed)
print(statistic)
print(p_value)

