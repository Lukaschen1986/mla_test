# -*- coding: utf-8 -*-
# https://www.bilibili.com/video/av37012291/?p=8
from flask import Flask
from flask_script import Manager # 命令行参数接收 shell, runserver

app = Flask(__name__)
manager = Manager(app=app)

@app.route("/demo")
def demo_func():
    return "Hello World!"

@app.route("/params/<idx>/")
def params_func(idx):
    print(idx)
    return "success"

@app.route("/get/<string:name>/")
def get_func(name):
    print(name)
    return "get success"





if __name__ == "__main__":
#    app.run(host="127.0.0.1", port=5000, debug=True)
    manager.run() # python demo_1.py runserver -d -r -h 127.0.0.1 -p 5000
