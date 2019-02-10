# -*- coding: utf-8 -*-
import pymysql
import config

class DbConnection(object):
    def __init__(self):
        self.__conn_dict = config.CONN_DICT
        self.conn = None
        self.cursor = None

    def connect(self, cur=pymysql.cursors.DictCursor):
        self.conn = pymysql.connect(**self.__conn_dict)
        self.cursor = self.conn.cursor(cursor=cur)
        return self.cursor

    def close(self):
        self.conn.commit()
        self.cursor.close()
        self.conn.close()
        return None
