import pandas as pd
import numpy as np
from dowhy import CausalModel
import warnings
import pydot
import pickle
import dowhy
import json
import dowhy.plotter

warnings.filterwarnings("ignore")
from datetime import datetime
import os
import re


class LazyDecoder(json.JSONDecoder):
    def decode(self, s, **kwargs):
        regex_replacements = [
            (re.compile(r"([^\\])\\([^\\])"), r"\1\\\\\2"),
            (re.compile(r",(\s*])"), r"\1"),
        ]
        for regex, replacement in regex_replacements:
            s = regex.sub(replacement, s)
        return super().decode(s, **kwargs)


class Testing:
    def __init__(self, data_path: str):
        self.binary_methods = [
            "backdoor.propensity_score_matching",
            "backdoor.propensity_score_weighting",
            "backdoor.propensity_score_stratification",
        ]
        self.data_path = data_path

    def get_input(self, config):
        # Objects

        # Data
        self.data_path = config["data"]["data_path"]
        self.target = config["data"]["target"]
        self.data_output_path = config["data"]["data_output_path"]
        self.model_output_path = config["data"]["model_output_path"]
        # Model
        self.version = config["model"]["version"]

        self.analyse = config["analyse"]

        artifact_file_path = os.path.join(
            self.data_output_path, f"{self.target}_artifacts.csv"
        )
        self.artifacts = pd.read_csv(artifact_file_path)

    def get_estimation_method(self):
        artifacts = pd.read_csv(
            f"{self.data_output_path}/artifacts_v{self.version}.csv"
        )

        version = self.version

        if version == -1:
            version = artifacts["Version"].values[-1]

        estimation_method = artifacts["EstimationMethod"].values[0]

        return estimation_method

    def get_treatements(self):

        treatments = self.artifacts[self.artifacts["Version"] == self.version]["TreatmentVariables"].values[0]
        
        treatments = treatments.split(",")
        treatments = [x.strip() for x in treatments]

        return treatments

    def read_data(self, target, treatment_vars):
        print("Reading data from GCS")

        data = pd.read_csv(f"{self.data_path}/predict_data.csv")
        data = data[treatment_vars] # + [target]]

        print(data.columns)
        print(len(data))
        return data

    def preprocess_data(self, df, target, treatment_vars):
        ## Clean the data

        # Replace null string values
        df.replace("#N/A", 0, inplace=True)
        df.replace("null", 0, inplace=True)

        # Replace infinte values
        df = df.drop_duplicates()

        # Fill null values by 0
        df = df.fillna(0)

        df[self.treatment_vars] = df[self.treatment_vars].astype(float)

        return df

    def fetch_model_version(self, target, model_output_path):
        model_versions = os.listdir(model_output_path[:-1])
        target_model_versions = pd.Series(model_versions).apply(
            lambda x: x if len(x.split(target)) > 1 else None
        )
        target_model_versions = list(
            target_model_versions[target_model_versions.notna()]
        )
        return target_model_versions

    def fetch_model(self):
        print("loading Model...")
        self.model_name = f"{self.target}_Causal_model_v{self.version}.pkl"
        model_path = os.path.join(self.model_output_path, self.model_name)
        model_file = open(model_path, "rb")
        causal_model = pickle.load(model_file)
        # close the file
        model_file.close()
        return causal_model

    def get_predictions(self, causal_model):
        print("Performing Prediction")
        x_dowhy_input = self.test_df[self.treatment_vars]
        x_dowhy_input.insert(0, "intercept", 1)

        y_pred_dowhy = causal_model.get_prediction(exog=x_dowhy_input.to_numpy())
        self.data[f"{self.target}_pred"] = y_pred_dowhy.predicted_mean.tolist()

        return self.data

    def save_pred_files(self):
        # Saving Predictions-------------------------------------------
        print("Saving Predictions...")

        filename = os.path.join(self.data_output_path, f"{self.model_name[:-4]}_predictions.csv")
        self.data.to_csv(
            filename,
            index=False,
        )

        return filename

    def predict(self, X, features_name=None):
        config = X[0]
        self.get_input(config)
        
        if self.data_path:
            print("true")
            # get treatment variables
            self.treatment_vars = self.get_treatements()

            # Read data from bigquery
            self.data = self.read_data(self.target, self.treatment_vars)

            # clean Data
            self.data = self.preprocess_data(
                self.data, self.target, self.treatment_vars
            )
            # load model
            causal_model = self.fetch_model()

            # estimation method
            # self.estimate_method = self.get_estimation_method()

            self.test_df = self.data.copy()

            print("columns order: ", self.test_df.columns)
            print("list order: ", self.treatment_vars)

        # predictions
        self.data = self.get_predictions(causal_model)

        # saving the model and model artifacts
        file_path = self.save_pred_files()

        target_var = self.data[f"{self.target}_pred"].tolist()

        print("Prediction Done")

        #         else :
        #             print('please provide test data to perform prediction')

        return {
            "output_path": file_path,
            #'target_var' : target_var,
            "treatment_vars": self.treatment_vars,
        }


# with open('C:\\Users\\pushpesh.pallav\\Downloads\\OneDrive_2023-12-11\\Causal as a Service Model\\Causal as a Service\\data\\input\\predict_config.json') as tc:
# #    config_load = user_file.read()
#     config = json.load(tc, cls=LazyDecoder)

# data_path = pathlib.Path(__file__).parent.joinpath("data")

# test = Testing()

# test.predict([config])
