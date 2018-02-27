# -*- coding: utf-8 -*-
import pandas as pd
from impala.dbapi import connect
from impala.util import as_pandas
import pymysql
from sqlalchemy import create_engine

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

engine = create_engine("mysql+pymysql://bds:pJdsS$00@10.250.33.163:3311/BigDataService?charset=utf8")
df.to_sql(name="t_api_callnum_alarm_2", con=engine, if_exists="append", index=False)
