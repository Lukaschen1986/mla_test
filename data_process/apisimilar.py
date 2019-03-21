# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import numpy as np


class ApiSimilar(object):
    def __init__(self, A, B):
        self.A = A
        self.B = B
    
    def bigrams(self, word):
        lst = []
        batchs = len(word) - 1
        for i in range(batchs):
            start = i
            end = start + 2
            lst.append(word[start:end])
        st = set(lst)
        return st
    
    def word_similar(self, word_1, word_2):
        assert isinstance(word_1, str), "word_1 must be astype: str"
        assert isinstance(word_2, str), "word_2 must be astype: str"
        a = self.bigrams(word_1)
        b = self.bigrams(word_2)
        res = 2 * len(a.intersection(b)) / (len(a)+len(b))
        return res
    
    def api_similar(self):
        # 计算A到B的相似度
        nmax_res = np.array([])
        for wa in A:
            sim_res = np.array([])
            for wb in B:
                sim = self.word_similar(wa, wb)
                sim_res = np.append(sim_res, sim)
            nmax = np.max(sim_res)
            nmax_res = np.append(nmax_res, nmax)
        navg_AB = np.mean(nmax_res)
        # 计算B到A的相似度
        nmax_res = np.array([])
        for wb in B:
            sim_res = np.array([])
            for wa in A:
                sim = self.word_similar(wb, wa)
                sim_res = np.append(sim_res, sim)
            nmax = np.max(sim_res)
            nmax_res = np.append(nmax_res, nmax)
        navg_BA = np.mean(nmax_res)
        # 计算相似度均值
        navg_res = (navg_AB + navg_BA) / 2
        # 计算惩罚因子：两个数组长度差异越大，惩罚越大，相似度越低
        punish = np.min([len(A),len(B)]) / np.max([len(A),len(B)])
        similar_res = navg_res * punish
        return similar_res


if __name__ == "__main__":
    A = ['str', 'stdfsdfasd', 'fsdfasdf']
    B = ['str', 'stdffasd', 'fsdfasd2f', 'str', 'stdfsfasd', 'fsdfsd2f']
    apisim = ApiSimilar(A, B)
    similar = apisim.api_similar()
    print(similar)
