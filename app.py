from flask import Flask, render_template, jsonify, request
from flask_cors import CORS, cross_origin
from train import train_data
from predict import predict_data
from logger import App_Logger
import pandas as pd
import pickle
import os

app = Flask(__name__)

log_writer = App_Logger()


@app.route('/', methods = ['GET'])
@cross_origin()
def home_page():
    return render_template('index.html')




@app.route('/predict', methods = ['GET', 'POST'])
@cross_origin()
def predict():
    if request.method == 'POST':
        try:
            file_object = open("logs/GeneralLogs.txt", 'a+')
            log_writer.log(file_object, 'Start getting data from UI')
            fixed_acidity = float(request.form['fixed_acidity'])
            volatile_acidity = float(request.form['volatile_acidity'])
            citric_acid = float(request.form['citric_acid'])
            residual_sugar = float(request.form['residual_sugar'])
            chlorides = float(request.form['chlorides'])
            free_sulfur_dioxide = float(request.form['free_sulfur_dioxide'])
            total_sulfur_dioxide = float(request.form['total_sulfur_dioxide'])
            density = float(request.form['density'])
            pH = float(request.form['pH'])
            sulphates = float(request.form['sulphates'])
            alcohol = float(request.form['alcohol'])

            log_writer.log(file_object, 'Complete getting data from UI')

            mydict = {'fixed acidity': fixed_acidity, 'volatile acidity': volatile_acidity, 'citric acid': citric_acid,
                      'residual sugar': residual_sugar, 'chlorides': chlorides, 'free sulfur dioxide': free_sulfur_dioxide,
                      'total sulfur dioxide': total_sulfur_dioxide, 'density': density, 'pH': pH,
                      'sulphates': sulphates, 'alcohol': alcohol}
            log_writer.log(file_object, 'Passing mydict to prediction.predict_data')
            prediction = predict_data(mydict, log_writer)

            return render_template('results.html', prediction=prediction)
            log_writer.log(file_object, '=================================================')
        except Exception as e:
            print('The Exception message is: ', e)
            return 'something is wrong'
            # return render_template('results.html')
    else:
        return render_template('index.html')

@app.route('/train', methods = ['GET', 'POST'])
@cross_origin()
def train():

    train_data(log_writer)


    return render_template('index.html')





if __name__ == "__main__":
    # clntApp = ClientApi()
    app.run(host='127.0.0.1', port=8001, debug=True)


