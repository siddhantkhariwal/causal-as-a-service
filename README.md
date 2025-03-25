# Causal As A Service
A causal model is a conceptual framework that describes the causal relationships between variables. It helps in understanding how changes in one variable might influence another. It is widely used to to analyze and predict the effects of interventions or changes in different factors.
## Index
- Overview
- Pre-requisites
- Understanding the Solution Architecture
- Understanding the Code
- Process
- Technologies Used

## Overview
This solution analyses the treatment and target variables present in data and returns the impact of each treatment variable on target variables in terms of causal estimate. The solution can be dynamically used for both Linear Regression and Generalized Linear Model (Logistic Regression) estimation methods.

## Pre-requisites
DoWhy is a Python library for causal inference that supports explicit modeling and testing of causal assumptions. DoWhy library serves are base for this solution and hence installing it is pre-requisite before running the python codes. This can be simply done by below command:

pip install dowhy

## Understanding the Solution Architecture
The solution consists of the below files:
1. Train_Causal_Model.ipynb     : Trains the model on basis of input data and saves the model as /model/model_name.pkl
2. Predict_Causal_Model.ipynb   : Predict's the target variable for the predict data as per the selected model
3. train_config.json            : Configuration file for Train_Causal_Model.ipynb
4. predict_config.json          : Predict_Causal_Model.ipynb
5. dag.txt                      : a text file containing graph structure describing relationship between variables

## Understanding the Code
A short description of function of each of the methods in the model is described below (in the same order of execution):
- get_input                 : reads the config file and initializes required model attributes
- read_dag_file             : reads the dag file
- read_data                 : reads the input data
- preprocess_data           : pre-processes input data
- get_treatments            : gets the list of all available treatment variables
- causal_estimate_logistic  : estimates causal estimate in-case of generalized linear model (logistic regression) estimation method is selected
- causal_estimate_linear    : estimates causal estimate in-case linear regression estimation method is selected
- refutation_check          : performs refutation
- predict                   : performs prediction after the model is trained
- save_model_files          : saves model output and artifacts in the specified output path
- causal_analysis           : main method that calls all the above methods

## Technologies Used
JUST python :)
Python Libraries used are - 
1. pandas
2. numpy
3. json
4. pickle
5. statsmodels.api


## Author
> Anuraag Bhavaraju
> Aman Singh Jakhra