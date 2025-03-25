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


class CustomerScenarios:
    def __init__(self, data_dir) -> None:
        self.data_dir = data_dir

    def get_input(self, config):
        # Data
        self.data_path = config["data"]["data_path"]
        self.target = config["data"]["target"]
        self.data_output_path = config["data"]["data_output_path"]
        self.model_output_path = config["data"]["model_output_path"]

        # self.fetch_models = config['fetch_models']

        # Model
        self.version = config["model"]["version"]

        self.new_data = config["new_data"]

        # old_data = config["old_data"]
        # if old_data != "":
        #     self.old_data = pd.DataFrame(old_data)

        artifact_file_path = os.path.join(
            self.data_dir, f"data/output/{self.target}_artifacts.csv"
        )
        self.artifacts = pd.read_csv(artifact_file_path)

        print("Target: ", self.target)

    def get_treatements(self):
        
        treatments = self.artifacts[self.artifacts["Version"] == self.version]["TreatmentVariables"].values[0]

        treatments = treatments.split(",")
        treatments = [x.strip() for x in treatments]

        return treatments

    def read_data(self, target, treatment_vars):
        print("Reading data from GCS")

        data = pd.read_csv(f"{self.data_dir}/data/input/predict_data.csv")
        data = data[treatment_vars + [target]]

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

    def fetch_model_and_coeffs(self, model_version):
        print("Loading Model...")
        self.model_name = f"{self.target}_Causal_model_v{model_version}.pkl"
        model_path = os.path.join(
            self.data_dir, self.model_output_path, self.model_name
        )
        model_file = open(model_path, "rb")
        causal_model = pd.read_pickle(model_file)
        # close the file
        model_file.close()

        coeffs = pd.read_csv(
            os.path.join(self.data_dir, f"data/output/{self.target}_coeffs.csv")
        )

        coeffs = coeffs[coeffs["Version"] == self.version]

        return causal_model, coeffs

    def get_predictions(self, df, causal_model):
        # print('Performing Prediction')
        x_dowhy_input = df[self.treatment_vars]
        x_dowhy_input.insert(0, "intercept", 1)

        y_pred_dowhy = causal_model.get_prediction(exog=x_dowhy_input.to_numpy())
        df[f"{self.target}_pred"] = y_pred_dowhy.predicted_mean.tolist()

        return df

    def get_feature_stats(self, treatment_vars, data):
        feature_stats = {}

        for col in treatment_vars:
            if data[col].nunique() == 2:
                col_type = "binary"
                min_value = data[col].min()
                max_value = data[col].max()
                mean = data[col].mean()
            else:
                col_type = "continuous"
                min_value = data[col].min()
                max_value = data[col].max()
                mean = data[col].mean()

            feature_stats[col] = {
                "col_type": col_type,
                "min_value": min_value,
                "max_value": max_value,
                "mean": mean,
            }

        return feature_stats

    def scenario_creation(self, X, analyze):
        config = X[0]
        self.get_input(config)

        # if fetch_models:
        #     #fetch the Trained Models
        #     model_version = self.fetch_model_version(self.target, self.model_output_path)
        #     print("model_versions: ", model_version[::-1])
        #     return model_version

        if analyze:
            self.treatment_vars = self.get_treatements()

            self.data = self.read_data(self.target, self.treatment_vars)

            self.data = self.preprocess_data(
                self.data, self.target, self.treatment_vars
            )

            causal_model, coeffs = self.fetch_model_and_coeffs(self.version)

            row_df = self.data.sample(n=1)

            # Prediction on first customer

            predictions = self.get_predictions(row_df, causal_model)

            predictions = predictions.to_dict()

            stats = self.get_feature_stats(self.treatment_vars, self.data)

            return {"predictions": predictions, "stats": stats, "coeffs": coeffs}

        else:
            self.data = self.read_data(self.target, self.treatment_vars)

            # clean Data
            self.data = self.preprocess_data(
                self.data, self.target, self.treatment_vars
            )

            # New Data
            new_data = pd.DataFrame(self.new_data)

            new_data = self.preprocess_data(new_data, self.target, self.treatment_vars)

            # load model
            causal_model, coeffs = self.fetch_model_and_coeffs(self.version)

            # estimation method
            # self.estimate_method = self.get_estimation_method()

            # predictions
            self.data = self.get_predictions(self.data, causal_model)

            # Prediction on new data
            predictions = self.get_predictions(new_data, causal_model)

            upper_caps = {x: self.data[x].quantile(0.95) for x in self.treatment_vars}

            for key, value in upper_caps.items():
                self.data = self.data[self.data[key] <= value]

            # return {
            #     'predictions': predictions,
            #     'data': self.data
            # }

            predictions.to_csv(
                os.path.join(
                    self.data_dir, "data/output/customer_output/new_predictions.csv"
                ),
                index=False,
            )

            self.data.to_csv(
                os.path.join(
                    self.data_dir, "data/output/customer_output/data_predictions.csv"
                ),
                index=False,
            )

            try:
                old_predictions = pd.read_csv(
                    os.path.join(
                        self.data_dir, "data/output/customer_output/old_predictions.csv"
                    )
                )

            except:
                old_predictions = pd.DataFrame()

            old_predictions = pd.concat([old_predictions, predictions])
            old_predictions.to_csv(
                os.path.join(
                    self.data_dir, "data/output/customer_output/old_predictions.csv"
                ),
                index=False,
            )
