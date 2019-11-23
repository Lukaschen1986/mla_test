# -*- coding: utf-8 -*-
from collections import deque
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
   
   
@get_time
def select_sort(lst):
    '''
    功能描述：选择排序
    参数：列表
    返回值：排序后列表（升序）
    异常描述：无
    修改记录：2019-11-23
    '''
    length = len(lst)
    
    # 外循环，控制轮次，一共 n-1 轮
    for i in range(0, length - 1):
        # 内循环，比较最小值
        for j in range(i + 1, length):
            if lst[i] > lst[j]:
                lst[i], lst[j] = lst[j], lst[i] # 变量交换
    
    return lst
    
lst = np.random.randint(low=0, high=100, size=20000).tolist()
lst_new = select_sort(lst) # select_sort 0 days 00:00:09.767887
