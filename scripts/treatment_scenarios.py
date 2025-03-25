import pandas as pd
import numpy as np
from dowhy import CausalModel
import warnings
import pydot
import pickle
import dowhy
import json
import dowhy.plotter
warnings.filterwarnings('ignore')
from datetime import datetime
import os
import re

class LazyDecoder(json.JSONDecoder):
    def decode(self, s, **kwargs):
        regex_replacements = [
            (re.compile(r'([^\\])\\([^\\])'), r'\1\\\\\2'),
            (re.compile(r',(\s*])'), r'\1'),
        ]
        for regex, replacement in regex_replacements:
            s = regex.sub(replacement, s)
        return super().decode(s, **kwargs)

class TreatmentScenarios:

    def __init__(self, data_dir) -> None:

        self.data_dir = data_dir

    def get_input(self, config):
        #Data
        self.data_path = config['data']['data_path']
        self.target = config['data']['target']
        self.data_output_path = config['data']['data_output_path']
        self.model_output_path = config['data']['model_output_path']

        print("what is target? ", self.target)
        #Model
        self.version = config['model']['version']

        self.tweak_col = config['data_change']['tweak_col']
        self.distribution = config['data_change']['distribution']
        self.min_value = config['data_change']['min_value']
        self.max_value = config['data_change']['max_value']
        self.mean = config['data_change']['mean']
        self.std_dev = config['data_change']['std_dev']

        artifact_file_path = os.path.join(
            self.data_dir, f"data/output/{self.target}_artifacts.csv"
        )
        self.artifacts = pd.read_csv(artifact_file_path)

    def get_treatements(self):
        
        treatments = self.artifacts[self.artifacts["Version"] == self.version]["TreatmentVariables"].values[0]
        
        treatments = treatments.split(',')
        treatments = [x.strip() for x in treatments]
        
        return treatments    
    
    def read_data(self, target, treatment_vars):
        
        print('Reading data from GCS')
        
        data = pd.read_csv(f"{self.data_dir}/data/input/predict_data.csv")
        data = data[treatment_vars + [target]]
        
        return data
    

    def preprocess_data(self, df, target, treatment_vars):
        ## Clean the data
        
        #Replace null string values
        df.replace('#N/A',0,inplace= True)
        df.replace('null',0,inplace= True)
        
        #Replace infinte values
        df = df.drop_duplicates()
        
        #Fill null values by 0
        df= df.fillna(0)
        
        df[self.treatment_vars] = df[self.treatment_vars].astype(float)

        return df  

    def fetch_model_version(self, target, model_output_path):
        model_versions = os.listdir(model_output_path[:-1])
        target_model_versions = pd.Series(model_versions).apply(lambda x:x if len(x.split(target)) > 1 else None)
        target_model_versions = list(target_model_versions[target_model_versions.notna()])
        return target_model_versions
    

    def fetch_model(self):

        print('loading Model...')
        self.model_name = f'{self.target}_Causal_model_v{self.version}.pkl'
        model_path = os.path.join(self.data_dir, self.model_output_path, self.model_name)
        model_file = open(model_path, 'rb')
        causal_model = pd.read_pickle(model_file)
        # close the file
        model_file.close()
        return causal_model

    def get_predictions(self, df, causal_model):
        print('Performing Prediction')
        x_dowhy_input = df[self.treatment_vars]
        x_dowhy_input.insert(0,'intercept',1)

        y_pred_dowhy = causal_model.get_prediction(exog = x_dowhy_input.to_numpy())
        df[f'{self.target}_pred'] = y_pred_dowhy.predicted_mean.tolist()

        return df
    
    def normal_distribution(self, size, mean, std_dev, min_value, max_value):
        
        np.random.seed(42)

        # Generate a normal distributed array
        normal_array = np.random.normal(loc=mean, scale=std_dev, size=size)

        # Clip the values to be within the specified range
        clipped_array = np.clip(normal_array, min_value, max_value)

        print("Clipped array mean: ", clipped_array.mean())
        return clipped_array
            

    def poisson_distribution(self, size, mean, min_value, max_value):
        while True:
            u = np.random.rand(size)

            arr = np.floor(np.log(u) / np.log(1 - 1/mean))
            arr = np.clip(arr, min_value, max_value)

            if np.all(arr >= min_value) and np.all(arr <= max_value):
                return arr
            
    def lognormal_distribution(self, size, mean, std_dev, min_value, max_value):
        while True:
            arr = np.random.lognormal(mean, std_dev, size)

            arr = np.clip(arr, min_value, max_value)

            if np.all(arr >= min_value) and np.all(arr <= max_value):
                return arr
            

    def scenario_creation(self, X, fetch_columns):

        config = X[0]
        self.get_input(config)

        # if fetch_models:
        #     #fetch the Trained Models
        #     model_version = self.fetch_model_version(self.target, self.model_output_path)
        #     print("model_versions: ", model_version[::-1])
        #     return model_version
    
        if fetch_columns:
            treatment_vars = self.get_treatements()
            return treatment_vars
        
        if self.data_path:
            self.treatment_vars = self.get_treatements()

            self.data = self.read_data( self.target, self.treatment_vars)

            #clean Data
            self.data = self.preprocess_data(self.data, self.target, self.treatment_vars)

            print("Data size:", len(self.data))
            #load model
            causal_model = self.fetch_model()

            #estimation method
            # self.estimate_method = self.get_estimation_method()

            new_df = self.data.copy()
            old_df = self.data.copy()

            var = self.tweak_col
            distribution = self.distribution
            min_value = float(self.min_value)
            max_value = float(self.max_value)
            mean = float(self.mean)
            std_dev = float(self.std_dev)
            size = len(self.data)

            # print("Mean: ", mean)

            if distribution.lower() == 'normal':
                new_df[var] = self.normal_distribution(size, mean, std_dev, min_value, max_value)

            if distribution.lower() == 'poisson':
                new_df[var] = self.poisson_distribution(size, mean, min_value, max_value)

            if distribution.lower() == 'lognormal':
                new_df[var] = self.lognormal_distribution(size, mean, std_dev, min_value, max_value)

            
            new_pred = self.get_predictions(new_df, causal_model)
            old_pred = self.get_predictions(old_df, causal_model)

            new_output = new_pred[[var, f"{self.target}_pred"]]
            old_output = old_pred[[var, f"{self.target}_pred"]]

            var_upper_cap = old_output[var].quantile(0.99)
            target_upper_cap = old_output[f"{self.target}_pred"].quantile(0.99)

            non_outlier_indices = old_output[(old_output[var] <= var_upper_cap) &
                                             (old_output[f"{self.target}_pred"] >= 0) &
                                             (old_output[f"{self.target}_pred"] <= target_upper_cap)].index
            

            old_output = old_output[old_output.index.isin(non_outlier_indices)]
            new_output = new_output[new_output.index.isin(non_outlier_indices)]

            old_output = old_output[[var, f"{self.target}_pred"]]
            new_output = new_output[[var, f"{self.target}_pred"]]
            
            # return {
            #     'old_df': old_df,
            #     'new_df': new_df
            # }

            # print(old_output.columns)
            # print(new_output.columns)
            old_output.to_csv(
                os.path.join(self.data_dir, 
                             "data/output/treatment_output/old_output.csv"),
                             index = False)
            
            new_output.to_csv(
                os.path.join(self.data_dir, 
                             "data/output/treatment_output/new_output.csv"), 
                             index = False)
