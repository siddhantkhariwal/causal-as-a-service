import os
import json
import streamlit as st
import pandas as pd


def render(version, target_variable, coefficents, causal_est_effect):
    st.markdown("<center><h1>Training Complete</h1></center>", unsafe_allow_html=True)

    st.markdown(f"<left><h4>Tareget Variable: {target_variable}</h4></left>", unsafe_allow_html=True)
    st.markdown(f"<left><h4>Model Version: {version}</h4></left>", unsafe_allow_html=True)
    st.markdown(f"<left><h4>Causal Estimate: {round(causal_est_effect, 2)}</h4></left>", unsafe_allow_html=True)

    coefficents = pd.DataFrame(coefficents)
    st.markdown(f"<left><h3>Coeffiecents </h3></left>", unsafe_allow_html=True)
    st.table(data=coefficents)
    


def train_widget(data_path: str, backend: "Training"):
    
    input_widget, output_widget = st.columns([0.25, 0.75])
    with input_widget:
        with st.container(border=True):
            st.markdown("### Data")
            data_object = st.file_uploader("Data File (CSV): ")
            target_variable = st.selectbox(
                "Target Variable: ", ["GMV_cur", "retention"]
            )
            dag_file = st.file_uploader("DAG File: ")

        with st.container(border=True):
            st.markdown("### Sampling")

            lower_cap = st.number_input("Target Lower Cap: ")
            upper_cap = st.number_input("Target Upper Cap: ", min_value=lower_cap)

        with st.container(border=True):
            st.markdown("### Methods")
            estimation_method = st.selectbox(
                "Estimation Method: ",
                ["backdoor.linear_regression", "backdoor.generalized_linear_model"],
            )
            refutation_method = st.selectbox(
                "Refutation Method: ",
                ["data_subset_refuter", "random_common_cause", None],
                index=2,
            )

            save = st.toggle("Save Model", value=True)

        if st.button("Train"):
            # Write uploaded files
            csv_path = os.path.join(data_path, "input", "input_data.csv")
            if data_object is not None:
                with open(csv_path, "wb") as f:
                    f.write(data_object.read())

            f_name = (
                "dag_linear.txt"
                if estimation_method == "backdoor.linear_regression"
                else "dag_glm.txt"
            )
            dag_path = os.path.join(data_path, "input", f_name)
            if dag_file is not None:
                with open(dag_path, "wb") as f:
                    f.write(dag_file.read())

            # Run Backend
            config = {
                "data": {
                    "target": target_variable,
                    "dag": "",
                    "dag_path": dag_path,
                    "target_upper_cap": upper_cap,
                    "target_lower_cap": lower_cap,
                    "data_path": csv_path,
                    "data_output_path": data_path.joinpath("output"),
                    "model_output_path": data_path.joinpath("..", "model"),
                },
                "methods": {
                    "estimation_method": estimation_method,
                    "refutation_method": refutation_method,
                },
                "model": {"save": save},
            }

            output = backend.predict([config])
            coefficents = output['coefficents']
            causal_est_effect = output['causal_estimate']

            version = output['version']
            with output_widget:
                render(version, target_variable, coefficents, causal_est_effect)
