import os
import json
import numpy as np
import pandas as pd
from glob import glob
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import re

bins = 70

def predict(
    backend, config: dict, output_widget, col: str, target: str, data_path: str
):
    config["data_change"]["tweak_col"] = col
    _ = backend.scenario_creation([config], fetch_columns=False)
    
    old = pd.read_csv(
        os.path.join(data_path, "output", "treatment_output", "old_output.csv")
    )

    new = pd.read_csv(
        os.path.join(data_path, "output", "treatment_output", "new_output.csv")
    )

    color_discrete_map = {"a": "blue", "b": "orange"}

    with output_widget:
        st.markdown("## New Variables")
        col_1, col_2 = st.columns(2)
        with col_1:
            st.markdown("### Treatment Variable")

            df = pd.DataFrame(
                dict(
                    series = np.concatenate((["a"]*len(old[col]), ["b"]*len(new[col]))),
                    data = np.concatenate((old[col], new[col]))
                )
            )


            plt = px.histogram(
                df,
                x = "data",
                color = "series", 
                nbins=bins, 
                barmode = "overlay", 
                color_discrete_map=color_discrete_map
                )

            st.plotly_chart(plt)
            st.markdown(
                f"### Mean: {round(new[col].mean(), 3)}\t\t\t Std Dev: {round(new[col].std(), 3)}"
            )
        with col_2:
            st.markdown("### Target Variable")

            df = pd.DataFrame(
                dict(
                    series = np.concatenate((["a"]*len(old[f"{target}_pred"]), ["b"]*len(new[f"{target}_pred"]))),
                    data = np.concatenate((old[f"{target}_pred"], new[f"{target}_pred"]))
                )
            )

            plt = px.histogram(
                df, x = "data", 
                color = "series", 
                nbins=bins, 
                barmode = "overlay",
                color_discrete_map=color_discrete_map

                )

            # fig = go.Figure()
            # fig = px.histogram(old[f"{target}_pred"], opacity = 0.7, nbins=30)
            # fig.add_histogram(new[f"{target}_pred"], opacity = 0.7, nbins=30)

            # fig.update_layout(title_text='Old vs New', barmode='overlay')
            st.plotly_chart(plt)
            st.markdown(
                f"### Mean: {round(new[f'{target}_pred'].mean(), 3)}\t\t\t Std Dev: {round(new[f'{target}_pred'].std(), 3)}"
            )


def analyze(
    backend, config: dict, output_widget, col: str, target: str, data_path: str
):
    config['target'] = target
    config["data_change"]["tweak_col"] = col
    _ = backend.scenario_creation([config], fetch_columns=False)
    old = pd.read_csv(
        os.path.join(data_path, "output", "treatment_output", "old_output.csv")
    )
    with output_widget:
        st.markdown("## Old Variables")
        col_1, col_2 = st.columns(2)
        with col_1:
            st.markdown("### Treatment Variable")
            plt = px.histogram(old[col], nbins = bins)
            st.plotly_chart(plt)
            st.markdown(
                f"### Mean: {round(old[col].mean(), 3)}\t\t\t Std Dev: {round(old[col].std(), 3)}"
            )
        with col_2:
            st.markdown("### Target Variable")
            plt = px.histogram(old[f"{target}_pred"], nbins = bins)
            st.plotly_chart(plt)
            st.markdown(
                f"### Mean: {round(old[f'{target}_pred'].mean(), 3)}\t\t\t Std Dev: {round(old[f'{target}_pred'].std(), 3)}"
            )

    

def treatment_widget(data_path: str, backend: "TreatmentScenarios"):
    input_widget, output_widget = st.columns([0.25, 0.75])
    with input_widget:
        with st.container(border=True):
            data_object = st.file_uploader("Data: ")
            if data_object is not None:
                df = pd.read_csv(data_object)
                df.to_csv(os.path.join(data_path, "input", "input_data.csv"))
                # with open(os.path.join(data_path, "input", "input_data.csv"), "w") as f:
                #     f.write(data_file.read())

            target = st.selectbox(
                "Target Variable: ", ["GMV_cur", "retention"], key="treatment_target"
            )

            # print("Target before analyze: ", target)
            

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
            model_version = st.selectbox("Select a model", sorted_models, index=sorted_models.index(latest_model))

            # model_version = st.selectbox("Model Version: ", models)
            st.button(
                "Fetch Columns",
                key="treatment_fetch_columns",
                on_click=lambda: st.session_state.update(
                    {"treatment_fetch_columns_active": True}
                ),
            )

        if "treatment_fetch_columns_active" not in st.session_state:
            st.session_state["treatment_fetch_columns_active"] = False
        if st.session_state["treatment_fetch_columns_active"]:
            with open(os.path.join(data_path, "input/treatment_config.json"), "r") as f:
                config = json.load(f)
                config["model"]["version"] = int(model_version.split("v")[-1])
                config['data']['target'] = target
            results = backend.scenario_creation([config], fetch_columns=True)
            with st.container(border=True):
                col_name = st.selectbox("Select Columns: ", results)
                st.button(
                    "Analyze",
                    key="treatment_analyze",
                    on_click=lambda: st.session_state.update(
                        {"treatment_analyze_active": True}
                    ),
                )

                if "treatment_analyze_active" not in st.session_state:
                    st.session_state["treatment_analyze_active"] = False
                if st.session_state["treatment_analyze_active"]:
                    config['data']['target'] = target
                    analyze(backend, config, output_widget, col_name, target, data_path)
                    # TODO draw distribution
                    with st.container(border=True):
                        distribution = st.selectbox(
                            "Distribution: ",
                            ["Normal", "Poisson", "Lognormal"],
                        )
                        min_value = st.number_input("Min Value: ")
                        max_value = st.number_input("Max Value: ", min_value=min_value)
                        mean = st.number_input("Mean: ")
                        std_dev = (
                            st.number_input("Std Dev: ")
                            if distribution in ["Normal", "Lognormal"]
                            else 1
                        )
                        if st.button("Predict", key="treatment_predict"):
                            config["target"] = target
                            config["data_change"] = {
                                "tweak_col": col_name,
                                "distribution": distribution,
                                "min_value": min_value,
                                "max_value": max_value,
                                "mean": mean,
                                "std_dev": std_dev,
                            }
                            config['data']['target'] = target
                            with output_widget:
                                predict(
                                    backend,
                                    config,
                                    output_widget,
                                    col_name,
                                    target,
                                    data_path,
                                )
