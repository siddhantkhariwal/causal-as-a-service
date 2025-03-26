import os
import json
from glob import glob
import streamlit as st
import pandas as pd
import re
import random

def analyze(data_path):
    st.session_state.update({"customer_analyze_active": True})
    try:
        os.remove(
            os.path.join(
                data_path,
                "output",
                "customer_output",
                "old_predictions.csv",
            )
        )
    except:
        pass


def customer_widget(data_path: str, backend: "CustomerScenarios"):
    input_widget, output_widget = st.columns([0.25, 0.75])
    with input_widget:
        data_object = st.file_uploader("Data (CSV): ", key="customer_data_file")
        if data_object is not None:
            df = pd.read_csv(data_object)
            df.to_csv(os.path.join(data_path, "input", "input_data.csv"))
            # with open(os.path.join(data_path, "input", "input_data.csv"), "w") as f:
            #     f.write(data_file.read())

        target = st.selectbox(
            "Target Variable: ", ["GMV_cur", "retention"], key="customer_target"
        )

        models = [
            f.split(os.path.sep)[-1].strip(".pkl")
            for f in glob(os.path.join(data_path, "..", "model", "*.pkl"))
            if f.split(os.path.sep)[-1].lower().startswith(target.lower())
        ]
        def extract_version(filename):
            match = re.search(r'v(\d+)', filename)
            return int(match.group(1)) if match else 0  # Default to 0 if no match

        # Sort models by version number
        sorted_models = sorted(models, key=extract_version, reverse=True)

        # Set the latest model as default
        latest_model = sorted_models[0] if sorted_models else None

        # model_version = st.selectbox("Model Version: ", models, key="customer_model")
        # Create a dropdown to select the model
        model_version = st.selectbox("Select a model", sorted_models, index=sorted_models.index(latest_model), key = "customer_model")

        st.button(
            "Analyze", key="customer_analyze", on_click=lambda: analyze(data_path)
        )
        if "customer_analyze_active" not in st.session_state:
            st.session_state["customer_analyze_active"] = False

        if st.session_state["customer_analyze_active"]:
            # TODO: Get columns
            with open(os.path.join(data_path, "input/customer_config.json"), "r") as f:
                config = json.load(f)
            config["model"]["version"] = int(model_version.split("v")[-1])
            config["data"]["target"] = target
            results = backend.scenario_creation([config], analyze=True)
            first_customer = pd.DataFrame(results["predictions"])

            first_customer = first_customer.round(2)

            coeffs = pd.DataFrame(results["coeffs"])

            
            stats = results["stats"]

            first_customer.to_csv(
                os.path.join(
                    data_path, "output", "customer_output", "first_customer.csv"
                ),
                index=False,
            )
            coeffs.to_csv(
                os.path.join(data_path, "output", "customer_output", "coeffs.csv"),
                index=False,
            )
            with open(
                os.path.join(data_path, "output", "customer_output", "stats.json"), "w"
            ) as f:
                json.dump(stats, f)

            # coeffs.rename({"Unnamed: 0": "Target Varible"}, axis = 1, inplace = True)
            
            with output_widget:
                st.markdown("### Coefficients")
                st.table(coeffs)
                st.markdown("### Base Customer")
                if "first_customer" not in st.session_state:
                    st.session_state.first_customer = first_customer

                st.table(st.session_state.first_customer)
                st.divider()

            values = {}

            if "slider_values" not in st.session_state:
                st.session_state.slider_values = {
                    col: random.uniform(stats[col]["min_value"], stats[col]["max_value"])
                    for col in stats
                }

            for col, stats in stats.items():
                values[col] = [
                    st.slider(
                        col, 
                        min_value=stats["min_value"],
                        max_value=stats["max_value"],
                        value=st.session_state.slider_values[col],
                    )
                ]
            if st.button("Predict"):
                with output_widget:
                    config["new_data"] = values
                    _ = backend.scenario_creation([config], analyze=False)
                    table = pd.read_csv(
                        os.path.join(
                            data_path,
                            "output",
                            "customer_output",
                            "old_predictions.csv",
                        )
                    )

                    st.markdown("### Predictions")
                    table = table.round(2)
                    st.table(table)