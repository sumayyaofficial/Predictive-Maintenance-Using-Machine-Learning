import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load saved objects
model = joblib.load("predictive maintenance.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("le.pkl")

st.set_page_config(
    page_title="Predictive Maintenance",
    layout="centered"
)

st.title("🛠️ Predictive Maintenance – Machine Failure Prediction")
st.write("Enter machine parameters to predict failure risk.")

st.divider()

# -----------------------------
# User Inputs (same order as training)
# -----------------------------
type_input = st.selectbox("Machine Type", ["L", "M", "H"])

air_temp = st.number_input(
    "Air Temperature [K]",
    min_value=290.0,
    max_value=320.0,
    value=300.0
)

process_temp = st.number_input(
    "Process Temperature [K]",
    min_value=300.0,
    max_value=350.0,
    value=310.0
)

rot_speed = st.number_input(
    "Rotational Speed [rpm]",
    min_value=1000,
    max_value=3000,
    value=1500
)

torque = st.number_input(
    "Torque [Nm]",
    min_value=0.0,
    max_value=100.0,
    value=40.0
)

tool_wear = st.number_input(
    "Tool Wear [min]",
    min_value=0,
    max_value=300,
    value=50
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Machine Failure"):

    # Encode Type
    type_encoded = le.transform([type_input])[0]

    # Create DataFrame
    input_df = pd.DataFrame({
    "Type": [type_encoded],
    "Air temperature [K]": [air_temp],
    "Process temperature [K]": [process_temp],
    "Rotational speed [rpm]": [rot_speed],
    "Torque [Nm]": [torque],
    "Tool wear [min]": [tool_wear],

    # Failure type indicators (unknown at prediction time → set to 0)
    "TWF": [0],
    "HDF": [0],
    "PWF": [0],
    "OSF": [0],
    "RNF": [0]
})

    # Scale numerical features
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.divider()

    if prediction == 1:
        st.error(
            f"⚠️ **Machine Failure Predicted**\n\n"
            f"Failure Probability: **{probability:.2%}**"
        )
    else:
        st.success(
            f"✅ **No Machine Failure Predicted**\n\n"
            f"Failure Probability: **{probability:.2%}**"
        )
