# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 00:11:42 2021

@author: ACER
"""

# pylint: disable=E1136
# pylint: disable=C0303

import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
import joblib as jb
import os


path = os.path.dirname(__file__)#r"C:/Users/ACER/Desktop/Workspace/spyder/Machine Learning/home price prediction project -1/datafiles/"

def df_desc(df_):
    """ Desc. of data set """
    print(df_.head(), "\n")
    print(df_.shape, "\n")
    print(df_.isnull().sum(), "\n")

def best_model(ind_var, target_var):
    """ Finding the best model and params """
    mods = {
        'linear_regression' : {
            'model' : LinearRegression(),
            'params' : {
                'normalize': [True, False]
                }
            }, 
        'lasso' : {
            'model' : Lasso(),
            'params' : {
                'alpha' : [1,3],
                'selection' : ['random', 'cyclic']
                }
            },
        'decision_tree' : {
            'model' : DecisionTreeRegressor(),
            'params' : {
                'criterion' : ['mse', 'friedman_mse'],
                'splitter' : ['best', 'random']
                }
            }
        }
    scores = []
    
    shuffle_split = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for mod_name, mod_config in mods.items():
        g_s_cv = GridSearchCV(mod_config['model'], mod_config['params'],
                              cv=shuffle_split, return_train_score=False)
        g_s_cv.fit(ind_var, target_var)
        scores.append({
            'model' : mod_name,
            'best_score' : g_s_cv.best_score_,
            'best_params' : g_s_cv.best_params_
            })
        return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])

def predict_function(location, sqft, bath, rooms):
    """ prediction test """
    loca_index = np.where(X.columns==location)[0][0]
    data = np.zeros(len(X.columns))
    
    data[0] = sqft
    data[1] = bath 
    data[2] = rooms
    
    if loca_index >= 0:
        data[loca_index] = 1
    
    return model1.predict([data])[0]
    
# reading the pre-processed dataset
df1 = pd.read_csv(f"{os.path.join(path,'datafiles/processed.csv')}")

# getting some insights of dataset
df_desc(df1)

# setting Independent variable
X = df1.drop(["price"], axis=1)

# setting the Depenent variable
y = df1["price"]

# splitting data set into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# creating the instance of linear reg. model
model1 = LinearRegression()

# training the Lr model on train dataset
model1.fit(X_train, y_train)

# getting the score of model on test dataset
print(f"score:\n{model1.score(X_test, y_test) * 100} % \n")

# shuffle and splitting dataset randomly
ShuffleSplit_val = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

# getting cross validation score from prev. shuffeled dataset
c_val_score = cross_val_score(LinearRegression(), X, y, cv=ShuffleSplit_val)

# printing the mean score of 5 iterations performed in above step
print(f"mean cross val score: \n{np.mean(c_val_score) * 100 } % \n")

# finding best model using GridSearchCV
print(f" The best model is: \n {best_model(X, y)}", "\n")
    
# testing and predicting some values
print(f"price = {round(predict_function('Indira Nagar',1600, 2, 2) * 100000)} rupees \n")

# saving the trained mode
os.path.join(path,'datafiles/')
MOD_NAME = os.path.join(path,'datafiles/model')
x = jb.dump(model1, MOD_NAME)
print(f"Model saved with name \"{MOD_NAME}\" \nsuccess!!! \n")

# loading the saved model
# saved_model = jb.load(MOD_NAME)

# getting all columns name in {data: [columns_name]} form
col = {"data" : [colm.lower() for colm in X.columns]}

# saving the above dict to json format
file = os.path.join(path,'datafiles/columns.json')
with open(file, "w") as f:
    f.write(json.dumps(col))
    