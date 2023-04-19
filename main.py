from flask import *
import pickle
import numpy as np
import pandas as pd
import sklearn
import re
import random
from random import *

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('attrition_prediction.html')

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,20)
    loaded_model = pickle.load(open("employee_attrition_mdl_1.pkl", "rb"))
    attrition = loaded_model.predict(to_predict)
    return attrition[0]


@app.route('/prediction', methods=['GET','POST'])
def prediction():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))

        if (len(to_predict_list)==20):
            attrition = ValuePredictor(to_predict_list)
            if (int(attrition) == 1):
                prediction = "Employee has a risk of Attrition"
            else:
                prediction = "Employee does't has a risk of Attrition"

        return (render_template('result.html', prediction_attrition = prediction))


if __name__ == "__main__":
    app.run(debug=True)