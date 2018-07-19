# -*- coding: utf-8 -*-
import pandas as pd
from impala.dbapi import connect
from impala.util import as_pandas
import pymysql
from sqlalchemy import create_engine, types

# Hive
class HiveClient(object):
    def __init__(self, host, user, password, database, port=10000, auth_mechanism="PLAIN"):
        self.conn = connect(host = host,
                            user = user,
                            password = password,
                            database = database,
                            port = port,
                            auth_mechanism = auth_mechanism)
    
    def query(self, sql):
        cursor = self.conn.cursor()
        cursor.execute(sql)
        return as_pandas(cursor)
    
    def close(self):
        self.conn.close()

interface = HiveClient(host="", user="", password="", database="interface", port=10000, auth_mechanism="PLAIN")

# MySQL
conn = pymysql.connect(host="", user="", passwd="", db="", charset="utf8")
conn.commit()
df = pd.read_sql(sql="", con=conn)
conn.close()

# 入库（删表重建）
engine = create_engine("mysql+pymysql://slusr:WW6LYvC6@10.171.199.172:3306/sl01?charset=utf8")
df_iair_historyprice.to_sql(name="t_iair_historyprice", con=engine, if_exists="replace", chunksize=None, 
                            index=True, index_label="RecordID",
                            dtype={"RecordID": types.INT(),
                                   "RangeType": types.VARCHAR(2),
                                   "DepatureCity3Code": types.VARCHAR(3),
                                   "ArrivalCity3Code": types.VARCHAR(3),
                                   "BunkGrade": types.VARCHAR(2),
                                   "UserCoverRate": types.VARCHAR(2),
                                   "HistoryPrice": types.FLOAT(precision=2),
                                   "CurrencyCode": types.VARCHAR(3),
                                   "CreateTime": types.DATETIME(),
                                   "LastModifyTime": types.TIMESTAMP()})

# 入库（不删表）
def insert_data(self, df_res, sql_truncate, sql_insert):
    isNaN = lambda x: x!=x
    # conn
    itravelhw = pymysql.connect(self.host, self.user, self.passwd, self.db, self.port, charset="utf8")
    itravelhw.commit()
    cur = itravelhw.cursor()
    # truncate exsit data
    cur.execute(sql_truncate)
    itravelhw.commit()
    # insert new data
    array_res = df_res.values
    t0 = pd.Timestamp.now()
    for i in range(len(array_res)):
        val = array_res[i].tolist()

        if isNaN(val[6]):
            val[6] = None

        cur.execute(sql_insert, val)
        itravelhw.commit()

    t1 = pd.Timestamp.now()
    cur.close()
    itravelhw.close()
    print("数据入库完毕，用时：%s" % str(t1-t0))
