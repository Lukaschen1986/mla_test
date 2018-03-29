    def data_load(self, sql):
        itravelhw = pymysql.connect(self.host, self.user, self.passwd, self.db, charset="utf8")
        itravelhw.commit()
    
        t0 = pd.Timestamp.now()
        dataSet = pd.read_sql(sql=sql, con=itravelhw)
        t1 = pd.Timestamp.now()
        itravelhw.close()
    
        dataSet.columns = dataSet.columns.str.lower()
        print("数据提取完毕，用时：%s" % (t1-t0))
        return dataSet
