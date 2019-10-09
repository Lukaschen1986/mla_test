# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 20:42:32 2019

@author: ASUS
"""

# 重复元素判定
def has_duplicates(lst):
    return len(lst) != len(set(lst))

# 判断两个字符串是否含有相同的元素
from collections import Counter
def anagram(str_1, str_2):
    return Counter(str_1) == Counter(str_2)

# 内存占用
import sys
variable = 2
print(sys.getsizeof(variable))

# 字节占用
def byte_size(string):
    return(len(string.encode("utf-8")))

# 大写所有字母
string = "programming is awesome"
string.upper()

# 大写第一个字母
string = "programming is awesome"
string.title()

# 小写所有字母
string = "PROGRAMMING IS AWESOME"
string.lower()

# 小写第一个字母
string = "PROGRAMMING"
def decapitalize(string):
    return string[0].lower() + string[1:]


# 分块
from math import ceil
def chunk(lst, size):
    func = lambda x: lst[x * size: x * size + size]
    res = list(
            map(
                    func, list(range(0, ceil(len(lst) / size)))
                    )
            )
    return res
#chunk([1,2,3,4,5],2)

# 逗号连接
hobbies = ["basketball", "football", "swimming"]
print("My hobbies are: " + ", ".join(hobbies))

# 元音统计
import re
def count_vowels(string):
    return len(len(re.findall(r"[aeiou]", string, re.IGNORECASE)))


# 元素频率
def most_frequent(lst):
    return max(set(lst), key = lst.count)

#lst = [1,2,1,2,3,2,1,4,2]


# 展开列表
def spread(lst):
    ret = []
    for i in lst:
        if isinstance(i, list):
            ret.extend(i)
        else:
            ret.append(i)
    return ret

#spread([1,2,3,[4,5,6],[7],8,9])
