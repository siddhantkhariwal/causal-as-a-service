import os
import pandas as pd
import numpy as np
from dowhy import CausalModel

from datetime import datetime as dt

# from mlutils import dataset, connector
pd.set_option("display.max_columns", 1000)
import warnings
import pydot
import pickle
import dowhy
import statsmodels.api as sm
# import mlflow
import json
import dowhy.plotter
from datetime import datetime
import json

warnings.filterwarnings("ignore")
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


class Training:
    def __init__(self, data_dir: str):
        self.version = 1
        self.binary_methods = [
            "backdoor.propensity_score_matching",
            "backdoor.propensity_score_weighting",
            "backdoor.propensity_score_stratification",
        ]
        self.data_path = data_dir

    def get_latest_version(self):
        try:
            artifacts = pd.read_csv(
                os.path.join(
                    self.data_output_path,
                    f"{self.target}_artifacts.csv"
                    )
            )

            latest_available_version = artifacts['Version'].values[-1]
        except:
            latest_available_version = 0

        return latest_available_version

    def get_treatments(self, text):
        treatments = text.split("\n")
        treatments = [x.strip(";") for x in treatments[1:-1] if "->" not in x]
        treatments.remove(self.target)
        return treatments

    def read_data(self, target, treatment_vars):
        print("Reading data from Input path")
        data = pd.read_csv(self.data_path)
        data = data[treatment_vars + [target]]
        return data

    def preprocess_data(self, df, target, treatment_vars):
        df.replace("#N/A", 0, inplace=True)
        df.replace("null", 0, inplace=True)
        # Replace infinite values
        df = df.replace([np.inf, -np.inf], 0)
        # Drop duplicates
        df = df.drop_duplicates()
        # Fill null values by 0
        df = df.fillna(0)
        df = df.astype(float)
        # df = df[(df[self.target] > self.lower_cap) & (df[self.target] > self.upper_cap)]
        print("Size of DF : ", len(df))
        return df

    def read_dag_file(self):
        print("Reading DAG file from Input path")
        with open(f"{self.dag_path}", "r") as f:
            dag = f.read()
        return dag

    def get_input(self, config):
        # objects
        # Data
        self.data_path = config["data"]["data_path"]
        self.dag = config["data"]["dag"]
        self.dag_path = config["data"]["dag_path"]
        self.data_output_path = config["data"]["data_output_path"]
        self.model_output_path = config["data"]["model_output_path"]
        self.target = config["data"]["target"]
        self.upper_cap = int(config["data"]["target_upper_cap"])
        self.lower_cap = int(config["data"]["target_lower_cap"])
        # Causal Methods
        self.estimate_method = config["methods"]["estimation_method"]
        self.refutation_method = config["methods"]["refutation_method"]
        # Read DAG File
        if self.dag == "":
            self.dag = self.read_dag_file()
        # Get treatment variables
        self.treatment_vars = self.get_treatments(self.dag)
        # Save Model
        self.save_status = config["model"]["save"]
        print("Treatment Variables: ", self.treatment_vars)
        print("Target: ", self.target)

    def causal_estimate_linear(self, model, identified_estimand):
        # Calcualte causal estimate using specific method
        print("Calculating the Causal estimate")
        print("method_name = ", self.estimate_method)
        causal_estimate = model.estimate_effect(
            identified_estimand, method_name=self.estimate_method, target_units="att"
        )
        # Causal Model
        causal_model = causal_estimate.estimator.model

        print(causal_model.summary())

        # Get Coefficients
        coeffs = causal_model.params
        coeffs_dict = dict(zip(["beta0"] + self.treatment_vars, coeffs))
        self.df_coeffs = pd.DataFrame(coeffs_dict, index=[self.target])
        return causal_model, self.df_coeffs, causal_estimate

    def causal_estimate_logistic(self, model, identified_estimand):
        method_params = {
            "num_null_simulations": 10,
            "num_simulations": 10,
            "num_quantiles_to_discretize_cont_cols": 10,
            "fit_method": "statsmodels",
            "glm_family": sm.families.Binomial(),
            "need_conditional_estimates": False,
        }
        estimate_method = "backdoor.genaralized_linear_model"
        estimates = {}
        causal_results = {}
        control_value = [0 for i in range(len(self.treatment_vars))]
        for i, v in enumerate(self.treatment_vars):

            treatment_value = [0 for i in range(len(self.treatment_vars))]
            treatment_value[i] = 1
            
            causal_estimate = model.estimate_effect(
                identified_estimand,
                method_name=self.estimate_method,
                test_significance=True,
                confidence_intervals=True,
                control_value=control_value,
                treatment_value=treatment_value,
                method_params=method_params,
            )

            estimates[v] = causal_estimate
            causal_model = causal_estimate.estimator.model
            causal_estimate.interpret(method_name="textual_effect_interpreter")
            causal_results[v] = causal_estimate.value
            break
        causal_model = estimates[list(estimates.keys())[0]].estimator.model
        coeffs = list(causal_model.params)
        coeffs_dict = dict(zip(["beta0"] + self.treatment_vars, coeffs))
        self.df_coeffs = pd.DataFrame(coeffs_dict, index=[self.target])
        return causal_model, self.df_coeffs, causal_estimate

    def refutation_check(self, model, identified_estimand, causal_estimate):
        
        if self.refutation_method == None:
            print("Skipping Refutation Check...")
            new_effect, p_value = 0, 0
        if self.refutation_method:
            print("Performing Refutation Check...")
            refutation_result = model.refute_estimate(
                identified_estimand, causal_estimate, method_name=self.refutation_method
            )
            new_effect = refutation_result.new_effect
            p_value = refutation_result.refutation_result["p_value"]
        else:
            print("Skipping Refutation Check...")
            new_effect, p_value = 0, 0

        return new_effect, p_value

    def causal_analysis(self):
        # Read data from GCS
        self.data = self.read_data(self.target, self.treatment_vars)
        
        # Clean Data
        self.data = self.preprocess_data(self.data, self.target, self.treatment_vars)

        train_df = self.data.copy()
        
        # Converting data to Binary if method selected is Binary method
        if self.estimate_method in self.binary_methods:
            for treatment in self.treatment_vars:
                # Convert the treatment variable to binary
                thresh = train_df[treatment].median()
                train_df[treatment] = train_df[treatment].apply(
                    lambda x: 1 if x > thresh else 0
                )
        print("Creating Causal Model")


        print(train_df.columns)
        
        # Step 1
        model = CausalModel(
            data=train_df,
            graph=self.dag,
            treatment=self.treatment_vars,
            outcome=self.target,
        )

        # Step 2
        print("Identifying the Causal effect")
        identified_estimand = model.identify_effect(
            proceed_when_unidentifiable=True, method_name="exhaustive-search"
        )

        # Step 3
        if self.estimate_method == "backdoor.linear_regression":
            # causal Estimate
            causal_model, df_coeffs, causal_estimate = self.causal_estimate_linear(
                model, identified_estimand
            )
            print("causal Estimate: ", causal_estimate.value)
        elif self.estimate_method == "backdoor.generalized_linear_model":
            # causal Estimate
            causal_model, df_coeffs, causal_estimate = self.causal_estimate_logistic(
                model, identified_estimand
            )
            print("causal Estimate: ", causal_estimate.value)

        # Step 4 Refutatoin Check
        new_effect, p_values = self.refutation_check(
            model, identified_estimand, causal_estimate
        )
        print("Training Completed for model")
        # Saving the model and model artifacts
        if self.save_status:
            self.save_model_files(
                causal_model,
                identified_estimand,
                df_coeffs,
                causal_estimate,
                new_effect,
                p_values,
            )
            print(f"Training Completed for model v{self.version}")
        else:
            print("Model not Saved!")
        return df_coeffs, causal_estimate.value

    def predict(self, X, features_name=None):
        config = X[0]
        print("Config: ", X)
        self.get_input(config)
        if self.data_path:
            df_coeffs, causal_est_effect = self.causal_analysis()
        else:
            print("Please provide training data!")
        return {
            "target": self.target,
            "version": self.version,
            "coefficents": df_coeffs.to_dict(),
            "causal_estimate": causal_est_effect,
        }

    def save_model_files(
        self,
        Causal_model,
        identified_estimand,
        df_coeffs,
        Causal_estimate,
        new_effect,
        p_value,
    ):
        # Saving Model--------------------------------------------------------------------------------
        # print('Registering the model...')

        try:
            self.version = self.get_latest_version() + 1
        except:
            print("First version of model is trained")
        self.model_name = os.path.join(
            self.model_output_path, f"{self.target}_Causal_model_v{self.version}.pkl"
        )
        with open(self.model_name, "wb") as f:
            pickle.dump(Causal_model, f)
        # Saving Model Artifacts----------------------------------------------------------------------
        print("Saving Model Artifacts...")

        try:
            artifacts = pd.read_csv(
                os.path.join(self.data_output_path,
                              f"{self.target}_artifacts.csv")
            )
        except:
            artifacts = pd.DataFrame(
                columns=[
                    "Target",
                    "TreatmentVariables",
                    "TableName",
                    "DAG",
                    "ModelName",
                    "CausalEstimate",
                    "RefutationNewEstimate",
                    "p_value",
                    "EstimationMethod",
                    "Version",
                    "ModelCreationDate",
                    "IdentifiedestimandFile",
                    "ModelURL",
                ]
            )

        artifacts_dict = {
            "Target": [self.target],
            "TreatmentVariables":[ ", ".join(self.treatment_vars)],
            "TableName": [self.data_path],
            "DAG":[ self.dag],
            "ModelName": [self.model_name],
            "CausalEstimate": [Causal_estimate.value],
            "RefutationNewEstimate": [new_effect],
            "p_value": [p_value],
            "EstimationMethod": [self.estimate_method],
            "Version": [self.version],
            "ModelCreationDate": [dt.today().date()],
            "IdentifiedEstimandFile": [f"{self.target}_identified_estimand_v{self.version}.pkl"],
            #'ModelURL': url,
        }

        # Updating the Artifact DataFrame
        artifacts = pd.concat([artifacts, pd.DataFrame.from_dict(artifacts_dict)], ignore_index=True) # artifacts.append(artifacts_dict, ignore_index=True)
        # Register the Artifact file to mlflow
        artifacts_file = os.path.join(self.data_output_path, f"{self.target}_artifacts.csv")
        artifacts.to_csv(artifacts_file, index=False)

        # Saving Identifies Estimand-------------------------------------------------------------------
        # print("Saving Identifies Estimand...")
        # artifacts_file = (
        #     os.path.join(self.data_output_path, f"{self.target}_identified_estimand_v{self.version}.pkl")
        # )

        # with open(artifacts_file, "wb") as f:
        #     pickle.dump(identified_estimand, f)
        
        # Saving Coeffs--------------------------------------------------------------------------------------------------------
        try:
            df_coeffs_db = pd.read_csv(os.path.join(self.data_output_path, f"{self.target}_coeffs.csv"))
        except:
            df_coeffs_db = pd.DataFrame()

        df_coeffs['Version'] = self.version
        df_coeffs['Target'] = self.target

        df_coeffs.reset_index(drop = True, inplace = True)
        
        print("Database:")
        print(df_coeffs_db)
        print()
        print("New Coeffs:")
        print(df_coeffs)
        df_coeffs_db = pd.concat([df_coeffs_db, df_coeffs])

        print()
        print("After merging:")
        print(df_coeffs_db)
        print("Saving Coefficents...")

        coeffs_file = os.path.join(self.data_output_path, f"{self.target}_coeffs.csv")
        df_coeffs_db.to_csv(coeffs_file, index=False)


# # Logic starts here
# with open(
#     "C:\\Users\\pushpesh.pallav\\Downloads\\OneDrive_2023-12-11\\Causal as a Service Model\\Causal as a Service\\data\input\\train_config_linear.json"
# ) as tc:
#     #    config_load = user_file.read()
#     config = json.load(tc, cls=LazyDecoder)

# train = Training()

# train.predict([config])
