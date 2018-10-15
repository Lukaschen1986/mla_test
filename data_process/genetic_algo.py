# -*- coding: utf-8 -*-
import os
os.getcwd()
import numpy as np
import pandas as pd
import copy
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import r2_score
#from scipy.stats import itemfreq
from tpot import TPOTRegressor
import matplotlib.pyplot as plt


def cal_time(func):
    def inner(*args, **kwargs):
        t0 = pd.Timestamp.now()
        res = func(*args, **kwargs)
        t1 = pd.Timestamp.now()
        t_delta = t1-t0
        print("Time cost {}".format(t_delta))
        return res
    return inner


class GeneticAlgorithm(object):
    def __init__(self, DNA_size, n_population, cross_rate, mutate_rate, steps):
        self.DNA_size = DNA_size
        self.n_population = n_population
        self.cross_rate = cross_rate
        self.mutate_rate = mutate_rate
        self.steps = steps
    
    # 种群函数
    def bulid_population(self):
        # 构建种群集合，行为种群总量，列位特征数量
        population = np.random.randint(low=0, high=2, size=(self.n_population, self.DNA_size)).astype(np.int8)
        row_sum = np.sum(population, axis=1)
        population = population[row_sum != 0]
        return population
    
    # 适应度函数：应算法选择而异
    def fitness(self, population, x_train, y_train, x_test, y_test):
        fitness_score = []
        
        for i in range(len(population)):
            # 筛选特征索引
            feats_idx = population[i].astype(np.bool)
            # 建子集
            x_train_sub = x_train.iloc[:,feats_idx]
            x_test_sub = x_test.iloc[:,feats_idx]
            # 模型训练 & 测试
            reg = linear_model.LinearRegression()
            reg.fit(x_train_sub, y_train)
            y_hat = reg.predict(x_test_sub)
            # 模型评估
            r2 = r2_score(y_test, y_hat)
            fitness_score.append(r2)
        # 调整 fitness_score
        fitness_score = np.array(fitness_score)
        fitness_score = np.where(fitness_score < 0, 0, fitness_score) # 针对r2的调整
        return fitness_score
    
    # 基因选择
    def select(self, population, fitness_score):
        n_population_local = len(population)
        a_population_local = np.arange(n_population_local)
        fitness_proba = fitness_score/fitness_score.sum()
        # 按优劣，有放回
        select_idx = np.random.choice(a=a_population_local, size=n_population_local, replace=True, p=fitness_proba)
        # 选出基因好的个体作为父代
        population = population[select_idx]
        return population
    
    # 基因交叉
    def crossover(self, parent, population_copy):
        # 如果满足 cross_rate
        if np.random.uniform() < self.cross_rate:
            # 随机选择一个优秀的父代进行繁衍
            idx = np.random.randint(low=0, high=len(population_copy))
            cross_sum = 0
            while cross_sum == 0:
                cross_points = np.random.randint(low=0, high=2, size=DNA_size).astype(np.bool) # 随机指定交叉位置
                cross_sum = np.sum(cross_points)
            parent[cross_points] = population_copy[idx, cross_points] # 赋值：繁衍
        return parent
    
    # 基因突变
    def mutate(self, offspring):
        for i in range(DNA_size):
            # 如果满足突变要求，则进行基因突变
            if np.random.uniform() < self.mutate_rate:
                # 如果原来是1，则变为0
                if offspring[i] == 1:
                    offspring[i] = 0
                # 如果原来是0，则变为1
                else:
                    offspring[i] = 1
        return offspring
    
    # 演化过程
    @cal_time
    def fit(self, x_train, y_train, x_test, y_test):
        # 第一步，初始化种群
        population = self.bulid_population()
        fitness_res = []
        
        for step in range(self.steps):
            # 第二步，适应度函数，计算种群中每个样本的适应度值(fitness_score)
            fitness_score = self.fitness(population, x_train, y_train, x_test, y_test)
            # 第三步，选择，进行自然选择，选出基因好的个体作为父代
            population = self.select(population, fitness_score)
            population_copy = copy.deepcopy(population)
            
            for parent in population:
                # 第四步，交叉，繁衍
                offspring = self.crossover(parent, population_copy)
                # 第五步，变异，基因突变
                offspring = self.mutate(offspring)
                # 将繁衍出的子代置为父代
                parent = offspring
            
            # 剔除全部为0的索引
            row_sum = np.sum(population, axis=1)
            population = population[row_sum != 0]
            # 保存
            fitness_mean = np.mean(fitness_score)
            fitness_res.append(fitness_mean)
            # 终止条件
            fitness_rollmean = pd.Series(fitness_res).rolling(20).mean()
            fitness_rollmean = fitness_rollmean.tolist()[-1]
            if fitness_mean <= fitness_rollmean: # 如果新的 fitness <= 移动平均值，则终止算法
                print("Break: fitness_mean <= fitness_rollmean, Steps: {}".format(step))
                break
        return population, fitness_res, fitness_score
    
    
if __name__ == "__main__":
    df = pd.read_csv("swiss.csv")
    df_train, df_test = train_test_split(df, test_size=0.3)
    
    x_train = df_train.iloc[:,0:5]
    y_train = df_train.iloc[:,5]
    x_test = df_test.iloc[:,0:5]
    y_test = df_test.iloc[:,5]

    _, DNA_size = x_train.shape
    
    # class
    GA = GeneticAlgorithm(DNA_size=DNA_size, n_population=300, cross_rate=0.9, mutate_rate=0.0, steps=100)
    population, fitness_res, fitness_score = GA.fit(x_train, y_train, x_test, y_test)
    plt.plot(fitness_res)
    
    # tpot
    tpot = TPOTRegressor(generations=100, population_size=300, crossover_rate=0.9, mutation_rate=0.0, cv=3, n_jobs=-1)
    tpot.fit(x_train, y_train)
    print(tpot.score(x_test, y_test))
    tpot.export('tpot_boston_pipeline.py')
        
