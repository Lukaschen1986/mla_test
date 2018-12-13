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

@app.route("/request/", methods=["POST","GET"])
def request_func():
#    print(request.args)
#    print(request.form)
#    print(request.data)
    print(request.args["name"])
    print(request.args.get("name"))
    print(request.args.getlist())
    
    name = request.args["name"]
    score_1 = request.args["score_1"]
    score_2 = request.args["score_2"]
    
    score_1 = np.float(score_1)
    score_2 = np.float(score_2)
#    
    score = score_1 + score_2
    res = jsonify(name=name, score=score) # Response
#    res = jsonify({"name": name, "score": score}) # Response
#    res = json.dumps({"name": name, "score": score}) # str
#    res = Response(response='{"name": name, "score": score}', content_type='application/json')
    return res

@app.route("/abort/")
def abort_func():
    abort(400)



if __name__ == "__main__":
#    app.run(host="127.0.0.1", port=5000, debug=True)
    manager.run() # python demo_1.py runserver -d -r -h 127.0.0.1 -p 5000
