# -*- coding: utf-8 -*-
# 迭代器:可以作用于for循环，也可以被next调用
from collections import Iterable, Iterator
l = [1,2,3,4,5]
isinstance(l, Iterable) # True 可迭代
isinstance(l, Iterator) # False 但不是迭代器

s = iter(l)
isinstance(s, Iterator) # True 是迭代器

#################################################################
# 生成器：一边循环，一边计算下一个元素，节省内存
import numpy as np
x = np.arange(100)
size = 10

def gen(x, size):
    times = len(x) // size
    for i in range(times):
        begin = size * i
        end = size + begin
        x_select = x[begin:end]
        yield x_select
    return "Done"

for i in gen(x, size):
    print(i)

#g = gen(x, size)
#for _ in range(10):
#    x = next(g)
#    print(x)
