# -*- coding: utf-8 -*-

from collections import (deque, Counter)
from functools import wraps
import numpy as np
import pandas as pd


def get_time(func):
    @wraps(func)
    def internal(*args, **kwargs):
        start = pd.Timestamp.now()
        res = func(*args, **kwargs)
        end = pd.Timestamp.now()
        print(func.__name__, end - start)
        return res
    return internal


# 顺序查找
@get_time
def sequence_search(lst, *, target):
    '''
    功能描述：顺序查找
    参数：列表，目标值（强制关键字，无默认值）
    返回值：第一个目标值的下标
    异常描述：无
    修改记录：2019-11-24
    '''
    for i in range(len(lst)):
        if lst[i] == target:
            return i
    return -1

lst = list(np.random.randint(low=0, high=100, size=50))
print(lst)

sequence_search(lst, target=83)
lst.index(83)
lst.index(83, 0, 30) # 返回在 [0,30) 的下标中第一个值是 83 的下标

# 计算值 83 出现的个数
lst.count(83)
c = Counter(lst)
c.get(83)


# 二分查找
@get_time
def binary_search(lst, *, target):
    '''
    功能描述：二分查找（前提：有序列表）
    参数：列表，目标值（强制关键字，无默认值）
    返回值：第一个目标值的下标
    异常描述：无
    修改记录：2019-11-24
    '''
    i_start = 0 # 第一位的下标
    i_end = len(lst) - 1 # 最后一位的下标
    
    while i_start <= i_end:
        i = (i_start + i_end) // 2
        
        if lst[i] == target:
            break
        elif target < lst[i]:
            i_end = i - 1
        else:
            i_start = i + 1
    
    if i_start <= i_end:
        return i
    else:
        return -1
    
lst = list(np.random.randint(low=10, high=1000, size=100))
lst = sorted(lst)
print(lst)

binary_search(lst, target=2)
