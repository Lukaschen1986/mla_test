# -*- coding: utf-8 -*-
# https://www.cnblogs.com/jinjidedale/p/6723725.html
from flask import Flask, abort, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

@app.route("/predict", methods=["POST","GET"])
def predict_func():
    # get params
    Agriculture = request.args["Agriculture"]
    Examination = request.args["Examination"]
    Education = request.args["Education"]
    Catholic = request.args["Catholic"]
    InfantMortality = request.args["InfantMortality"]
    print(Agriculture, Examination, Education, Catholic, InfantMortality)
    # modify type
    Agriculture = np.float64(Agriculture)
    Examination = np.int64(Examination)
    Education = np.int64(Education)
    Catholic = np.float64(Catholic)
    InfantMortality = np.float64(InfantMortality)
    # DataFrame
    x_predict = pd.DataFrame({"Agriculture": [Agriculture],
                              "Examination": [Examination],
                              "Education": [Education],
                              "Catholic": [Catholic],
                              "InfantMortality": [InfantMortality]}, columns=["Agriculture","Examination","Education","Catholic","InfantMortality"])
    # data process
    txt_path = "D:/my_project/Python_Project/iTravel/flask/txt/"
    with open(txt_path + "process_fit.txt", "rb") as f:
        process_fit = pickle.load(f)
    x_pred_new = process_fit.transform(x_predict)
    # data predict
    with open(txt_path + "clf_fit.txt", "rb") as f:
        clf_fit = pickle.load(f)
    
    y_hat = clf_fit.predict_proba(x_pred_new)
    y_pred = np.argmax(y_hat, axis=1)
    y_pred = np.float64(y_pred[0])
    print("y_pred: ", y_pred)
    # response
    res = jsonify(y_pred=y_pred)
    return res


if __name__ == "__main__":
    # 将host设置为0.0.0.0，则外网用户也可以访问到这个服务
    app.run(host="127.0.0.1", port=5000, debug=False)



