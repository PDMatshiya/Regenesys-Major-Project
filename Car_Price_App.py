#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 28 19:00:54 2025

@author: pdmatshiya
"""
# Section 8: Model Deployment with web app

# 8.1: Develop an interactive web application using Streamlit.
import streamlit as st
import pickle
import numpy as np
from fpdf import FPDF
import time


# Load trained Decision Tree model
with open("decision_tree_model.pkl", "rb") as f:
    model = pickle.load(f)


# Initialize session state
if 'vehicle_age' not in st.session_state:
    st.session_state.vehicle_age = 5
if 'max_power' not in st.session_state:
    st.session_state.max_power = 80.0
if 'engine' not in st.session_state:
    st.session_state.engine = 1500
if 'fuel_type_diesel' not in st.session_state:
    st.session_state.fuel_type_diesel = 0
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

# Create function that will create a pdf where users can extract car price prediction information
def create_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Car Price Prediction", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Vehicle Age: {st.session_state.vehicle_age} years", ln=True)
    pdf.cell(0, 8, f"Max Power: {st.session_state.max_power} bhp", ln=True)
    pdf.cell(0, 8, f"Engine Size: {st.session_state.engine} cc", ln=True)
    pdf.cell(0, 8, f"Fuel Type: {'Diesel' if st.session_state.fuel_type_diesel else 'Other'}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, f"Predicted Selling Price: R{st.session_state.prediction:,.2f}", ln=True)
    
    return bytes(pdf.output(dest='S'))


# Create App User Interface
st.title("🚗 Car Price Prediction App")
st.markdown("Fill in the details below to get an estimate of your car's selling price.")

# Manual number inputs with min/max info
st.session_state.vehicle_age = st.number_input(
    "Vehicle Age (years) [Min: 0, Max: 14]", min_value=0, max_value=14, value=st.session_state.vehicle_age
)

st.session_state.max_power = st.number_input(
    "Max Power (bhp) [Min: 30, Max: 200]", min_value=30.0, max_value=200.0, value=st.session_state.max_power, step=1.0
)

st.session_state.engine = st.number_input(
    "Engine Size (cc) [Min: 500, Max: 2500]", min_value=500, max_value=2500, value=st.session_state.engine, step=50
)

is_diesel = st.radio(
    "Is the fuel type Diesel?", ("No", "Yes"),
    index=st.session_state.fuel_type_diesel
)
st.session_state.fuel_type_diesel = 1 if is_diesel == "Yes" else 0

# Prediction button
if st.button("Predict"):
    features = np.array([[st.session_state.vehicle_age,
                          st.session_state.max_power,
                          st.session_state.engine,
                          st.session_state.fuel_type_diesel]])
    
    # Loading spinner
    with st.spinner("Predicting..."):
        time.sleep(1)
        prediction = model.predict(features)[0]
        st.session_state.prediction = prediction

    # Animated metric counting up
    placeholder = st.empty()
    predicted_value = 0
    step_value = max(prediction / 100, 100)
    while predicted_value < prediction:
        predicted_value += step_value
        if predicted_value > prediction:
            predicted_value = prediction
        placeholder.metric("Predicted Selling Price (R)", f"{predicted_value:,.2f}")
        time.sleep(0.01)

    # Insight message
    st.info(
        "💡 Insight: While higher vehicle age usually lowers the price, a more powerful engine or larger max power can compensate. "
        "This prediction reflects the combination of your car's features."
    )

    # Download PDF button
    pdf_bytes = create_pdf()
    st.download_button(
        label="📄 Download Prediction as PDF",
        data=pdf_bytes,
        file_name="car_price_prediction.pdf",
        mime="application/pdf"
    )











