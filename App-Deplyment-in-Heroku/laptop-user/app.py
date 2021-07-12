# creating a Flask app
from flask import Flask, jsonify, request, render_template
import pickle
import pandas as pd
import numpy as np
app = Flask (__name__)


@app.route('/')
def home():
    return render_template("index.html")



#prediction function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,12)
    loaded_model = pickle.load(open("model.pkl","rb"))
    result =  loaded_model.predict(to_predict)
    return result[0]

@app.route('/index', methods = ['POST'])
def result():
    if request.method== 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list)
        if int(result) ==1:
            prediction = "Yes"
        else:
            prediction = "No"
        return render_template("index.html", prediction = prediction)


if __name__ == '__main__':
    app.run(debug=True)
    app.config["TEMPLATES_AUTO_RELOAD"]
    
