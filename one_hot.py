one_hot = lambda y: np.eye(len(set(y)))[y]

X,Y = ['\u4e00','\u9fa5']
X
Y
X<='jame'<=Y

df.index[0].year
df.index[0].month
df.index[0].day
df.index[0].weekday()

df[col].plot(grid=True, figsize=(10.8,7.6))

compare = pd.DataFrame({"plot_1": df[col_1],
                        "plot_2": df[col_2]})
compare.plot(grid=True, figsize=(10.8,7.6))

func = lambda x: np.max(x)-np.min(x)
df.groupby(key).transform(func)
df.groupby(key).transform("max") - df.groupby(key).transform("min")


res = pd.concat((df1,df2,df3), keys=["1","2","3"]) # 追加一列新的index
res.loc["1"]


pd.concat((df1,df2,df3), axis=1, join="inner")
pd.concat((df1,df2,df3), axis=1, join_axes=[df1.index])

df_order_full.columns = df_order_full.columns.str.replace("orderid","ordernum") # 列名修改
df_order_full.index = df_order_full.reset_index().rename(columns={"orderid","ordernum"}) # 索引名修改

df = pd.merge(df1, df2, on="...", how="inner")
df1.join(df2, how="inner")

df["col"].first_valid_index() # 列中第一个不为空的index
index > df["idx"].first_valid_index()
