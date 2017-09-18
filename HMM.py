# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

A = pd.DataFrame({"sunny": [0.5,0.25,0.25],
                  "cloudy": [0.375,0.125,0.375],
                  "rainy": [0.125,0.25,0.5]}, index=["cloudy","rainy","sunny"])

B = pd.DataFrame({"dry": [0.6,0.25,0.05],
                  "dryish": [0.2,0.25,0.1],
                  "damp": [0.15,0.25,0.35],
                  "soggy": [0.05,0.25,0.5]}, index=["cloudy","rainy","sunny"])

pi = pd.DataFrame({"sunny":[0.6], "cloudy":[0.2], "rainy":[0.2]})

obs = ('dry','dryish','damp','soggy')

n = len(A)
t = len(obs)

# 1
x1 = pi.dot(A)

full_prob = 0.0
for i in range(n):
    full_prob += B.loc[x1.columns[i], obs[0]]*x1.iloc[:,i]

update = np.zeros((1,n))
for j in range(n):
    update[0,j] = B.loc[x1.columns[j], obs[0]]*x1.iloc[:,j] / full_prob
update = pd.DataFrame(update, columns=A.columns)


# 2
predict = update.dot(A)

full_prob = 0.0
for i in range(n):
    full_prob += B.loc[predict.columns[i], obs[1]]*predict.iloc[:,i]
update = np.zeros((1,n))
for j in range(n):
    update[0,j] = B.loc[predict.columns[j], obs[1]]*predict.iloc[:,j] / full_prob
update = pd.DataFrame(update, columns=A.columns)

# 3
predict = update.dot(A)

full_prob = 0.0
for i in range(n):
    full_prob += B.loc[predict.columns[i], obs[2]]*predict.iloc[:,i]
update = np.zeros((1,n))
for j in range(n):
    update[0,j] = B.loc[predict.columns[j], obs[2]]*predict.iloc[:,j] / full_prob
update = pd.DataFrame(update, columns=A.columns)

# 4
predict = update.dot(A)

full_prob = 0.0
for i in range(n):
    full_prob += B.loc[predict.columns[i], obs[3]]*predict.iloc[:,i]
update = np.zeros((1,n))
for j in range(n):
    update[0,j] = B.loc[predict.columns[j], obs[3]]*predict.iloc[:,j] / full_prob
update = pd.DataFrame(update, columns=A.columns)

# 5
predict = update.dot(A)
np.argmax(np.array(predict), axis=1)
predict.columns[np.argmax(np.array(predict))]

-----------------------

df = pd.read_csv("seatbelts.csv")
idx = np.random.permutation(df.index)
df = pd.DataFrame(df, index=idx)
df.index = range(len(df))

df = df[0:20]


x = df["price"]
A_name = set(x)
A_shape = len(A_name)
A = pd.DataFrame(np.zeros((A_shape, A_shape)), index=A_name, columns=A_name)

for i in range(len(x)-1):
    for j in range(1,len(x)):
#        i = 9; j = 10
        A.loc[x[i], x[j]] = A.loc[x[i], x[j]] + 1

A_sum = A.apply(np.sum, axis=1)
for j in range(len(A)):
    A.iloc[j,:] = A.iloc[j,:] / A_sum[j]

----------------------------------------------
df <- data.frame(Seatbelts[,c("DriversKilled","PetrolPrice")])
kill <- cut(df$DriversKilled, breaks = 5, right = F, include.lowest = T)
price <- cut(df$PetrolPrice, breaks = 4, right = F, include.lowest = T)
df <- data.frame(kill = kill, price = price)
write.csv(df, "D:/my_project/Python_Project/test/HMM/seatbelts.csv", row.names = F)
