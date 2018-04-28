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
