import os
import re
import pandas as pd
import streamlit as st
from glob import glob
import plotly.express as px
import plotly.graph_objects as go


def render(output_path, predictions, treatment_vars, target_variable):

    treatment_vars = ", ".join(treatment_vars)
    st.markdown("<center><h1>Prediction Complete</h1></center>", unsafe_allow_html=True)
    st.markdown(f"<left><h4>Treament Variables: </h4> <p>{treatment_vars}</p></left>", unsafe_allow_html=True)
    st.markdown(f"<left><h4>Predictions are saved to: </h4><p>{output_path}</p></left>", unsafe_allow_html=True)

    

    st.markdown("<left><h4>Prediction Distribution<h4></left>", unsafe_allow_html=True)

    plt = px.histogram(
                predictions,
                x = f"{target_variable}_pred",
                nbins=40
                )
    
    st.plotly_chart(plt)



def predict_widget(data_path: str, backend: "Testing"):
    input_widget, output_widget = st.columns([0.25, 0.75])
    with input_widget:
        with st.container(border=True):
            st.markdown("### Data")
            data_object = st.file_uploader("Data File (CSV): ", key="predict_data")
            if data_object is not None:
                df = pd.read_csv(data_object)
                df.to_csv(os.path.join(data_path, "input", "input_data.csv"))
            # with open(os.path.join(data_path, "input", "input_data.csv"), "w") as f:
            #     f.write(data_file.read())

            target_variable = st.selectbox(
                "Target Variable: ", ["GMV_cur", "retention"], key="predict_target"
            )

            with st.container(border=True):
                st.markdown("### Model")
                models = [
                    f.split(os.path.sep)[-1].strip(".pkl")
                    for f in glob(os.path.join(data_path, "..", "model", "*.pkl"))
                    if f.split(os.path.sep)[-1]
                    .lower()
                    .startswith(target_variable.lower())
                ]
                version = st.selectbox(
                    "Model Version: ", options=models, key="predict_model"
                )

                
                if st.button("Predict", key="Predict_button"):
                    # print("Model Version: ", int(version.strip()[-1]))
                    model = int(re.findall("\d+", version)[0])
                    print("Model Version: ", model)
                    config = {
                        "data": {
                            "target": target_variable,
                            "data_path": data_path.joinpath("input"),
                            "data_output_path": data_path.joinpath("output"),
                            "model_output_path": data_path.joinpath("..", "model"),
                        },
                        "model": {"version": model},
                        "analyse": True,
                    }
                    output = backend.predict([config])
                    output_path = output["output_path"]
                    treatment_vars = output["treatment_vars"]

                    predictions = pd.read_csv(output_path)

                    with output_widget:
                        render(output_path, predictions, treatment_vars, target_variable)
