# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 18:32:43 2021

@author: ACER
"""

# pylint: disable=E1136
# pylint: disable=C0303

import json
import time
import threading as t
import webbrowser
import joblib as jb
import numpy as np
from flask import Flask, request, jsonify, render_template

__locations = None
__data_columns = None
__model = None

path = (r"../datafiles/")

browser_address = ("C://Users//ACER//AppData//Local//Google//Chrome" \
                   "//Application//chrome.exe %s")

def open_browser():
    """ Opens browser automatically after 10 seconds """
    time.sleep(10)
    webbrowser.get(browser_address).open("http://127.0.0.1:5000/")

def get_est_price(location, sqft, rooms, bath):
    """ Try to predict the price """

    try:
        loc_index = __data_columns.index(location.lower())

    except IndexError:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = rooms

    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0],2)

def load_saved_artifacts():
    """ Loads the saved data files """

    print("loading saved artifacts...start")
    global  __data_columns
    global __locations

    with open(f"{path + 'columns.json'}", "r") as f:
        __data_columns = json.load(f)["data"]
        __locations = __data_columns[3:]  # first 3 columns are sqft, bath, bhk

    global __model
    if __model is None:
        with open(f"{path + 'model'}", 'rb') as f:
            __model = jb.load(f)
    print("loading saved artifacts...done")


app = Flask(__name__)


@app.route('/test')
def test_page():
    """ Its test page to check if server is Up """
    return  render_template("testpage.html")


@app.route('/')
def main_app():
    """ Main page of Flask App """
    return  render_template("app.html")


@app.route('/get_location_names', methods=['GET'])
def get_location_names():
    """ Getting all loction names from saved data """
    response = jsonify(
            {
                'locations': __locations
            }
        )
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/predict_home_price', methods=['GET', 'POST'])
def predict_home_price():
    """ Function to predict the price of house """
    total_sqft = float(request.form['total_sqft'])
    location = request.form['location']
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])

    response = jsonify({
        'estimated_price': get_est_price(location,total_sqft,bhk,bath)
    })

    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == "__main__":

    print("Starting Python Flask Server For Home Price Prediction...")
    load_saved_artifacts()
    time.sleep(2)
    t.Thread(target=open_browser).start()
    time.sleep(2)
    app.run(debug=True)
