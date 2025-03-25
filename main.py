import pathlib
import streamlit as st

from ui import train_widget, predict_widget, treatment_widget, customer_widget
from scripts import Training, Testing, TreatmentScenarios, CustomerScenarios


st.set_page_config(layout="wide")

DATA_PATH = pathlib.Path(__file__).parent.joinpath("data")
print(DATA_PATH)

DATA_PATH.joinpath("input").mkdir(exist_ok=True)
DATA_PATH.joinpath("output").mkdir(exist_ok=True)


def main():
    st.markdown("<h2 style='text-align: center;'>Causal Analysis</h2>"
    , unsafe_allow_html=True)

    # Display the formula centered
    st.markdown(
        "<h5 style='text-align: center;'>CX Score = GMV * Customer Retention</h5>",
        unsafe_allow_html=True
    )

    train, predict, treatment, customer = st.tabs(
        ["Train", "Predict", "Treatment Scenarios", "Customer Scenarios"]
    )

    backend = {
        "train": Training(DATA_PATH),
        "predict": Testing(DATA_PATH),
        "treatment": TreatmentScenarios(DATA_PATH.parent),
        "customer": CustomerScenarios(DATA_PATH.parent),
    }

    with train:
        train_widget(DATA_PATH, backend["train"])

    with predict:
        predict_widget(DATA_PATH, backend["predict"])

    with treatment:
        treatment_widget(DATA_PATH, backend["treatment"])

    with customer:
        customer_widget(DATA_PATH, backend["customer"])


main()
