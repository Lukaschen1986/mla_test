# -*- coding: utf-8 -*-
from .db import DbConnection
import time

class DiaryModel(DbConnection):
    def create_diary(self, **kwargs):
        try:
            self.connect()
            sql = "insert into diary (weather, mood, content, c_time) values (%s, %s, %s, %s)"
            self.cursor.execute(sql, (kwargs["weather"], kwargs["mood"], kwargs["content"], time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())))
            self.close()
            return self.cursor.lastrowid
        except Exception as e:
            print(e)

    def get_info_by_id(self, diary_id):
        try:
            self.connect()
            sql = "select * from diary where id = %s"
            self.cursor.execute(sql, (diary_id, ))
            self.close()
            return self.cursor.fetchone()
        except Exception as e:
            print(e)

    def get_diary_list(self):
        self.connect()
        sql = "select * from diary"
        try:
            self.cursor.execute(sql)
            self.close()
            return self.cursor.fetchall()
        except Exception as e:
            print(e)



