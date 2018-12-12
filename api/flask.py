# -*- coding: utf-8 -*-
# https://www.cnblogs.com/jinjidedale/p/6723725.html
from flask import Flask, abort, request, jsonify
import pickle
import numpy as np
import pandas as pd
import pymysql

app = Flask(__name__)

@app.route("/add", methods=["POST"])
def add_func():
    # conn
    conn = pymysql.connect(host="127.0.0.1", user="root", passwd="chen1986", db="test", port=3306, charset="utf8")
    cur = conn.cursor()
    # add_info
    add_info = {
            "MemberID": request.json["MemberID"],
            "UserName": request.json["UserName"],
            "Age": request.json["Age"]
            }
    sql = "INSERT INTO t_test VALUES({0},{1},{2})".format(add_info["MemberID"], add_info["UserName"], add_info["Age"])
    # execute
    cur.execute(sql)
    conn.commit()
    print(sql)
    # close
    conn.close()
    return "insert data success"
    

@app.route('/<int:id>', methods=['GET'])
def query(id):
    # 连接数据库
    conn = pymysql.connect(host="127.0.0.1", user="root", passwd="chen1986", db="test", port=3306, charset="utf8")
    cur = conn.cursor()
    # 执行sql语句
    sql = "select MemberID,UserName,Age from t_test where MemberID = " + str(id)
    cur.execute(sql)
    result = cur.fetchall()
    print(sql)
    # 关闭连接
    conn.close()
    return jsonify(
         {
             'MemberID': result[0][0],
             'UserName': result[0][1],
             'Age': result[0][2],
         })


@app.errorhandler(404)
def page_not_found(e):
    res = jsonify({'error': 'not found'})
    res.status_code = 404
    return res

#DROP TABLE IF NOT EXISTS t_test;
#CREATE TABLE t_test (
#	MemberID INT UNSIGNED NOT NULL auto_increment COMMENT '用户ID',
#	UserName VARCHAR(20) NOT NULL COMMENT '用户名',
#	Age INT UNSIGNED NOT NULL COMMENT '年龄',
#	PRIMARY KEY(MemberID)
#);
#SELECT * FROM t_test;

if __name__ == "__main__":
    # 将host设置为0.0.0.0，则外网用户也可以访问到这个服务
    app.run(host="127.0.0.1", port=5000, debug=True)





