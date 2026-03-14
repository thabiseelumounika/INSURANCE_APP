import streamlit as st
import pandas as pd
import joblib   #parameters read 

# Load saved objects
model = joblib.load("LR_model.pkl")
encoder = joblib.load("onehotencoder.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Insurance Expense Prediction App")

st.write("Enter the details below to predict medical insurance expenses.")

# User Inputs
age = st.number_input("Age", min_value=18, max_value=100, step=1)

sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

bmi = st.number_input("BMI", min_value=10.0, max_value=60.0)

children = st.number_input("Number of Children", min_value=0, max_value=10, step=1)

smoker = st.selectbox(
    "Smoker",
    ["yes", "no"]
)

region = st.selectbox(
    "Region",
    ["northeast", "northwest", "southeast", "southwest"]
)

# Prediction Button
if st.button("Predict Insurance Cost"):

    # Create dataframe from inputs
    new_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })

    # Categorical columns
    cat_cols = ['sex', 'smoker', 'region']

    # Encode categorical data
    encoded = encoder.transform(new_data[cat_cols])

    encoded_cols = encoder.get_feature_names_out(cat_cols)

    encoded_df = pd.DataFrame(encoded, columns=encoded_cols)

    # Combine with numeric features
    final_data = pd.concat([new_data.drop(columns=cat_cols), encoded_df], axis=1)

    # Scale data
    scaled_data = scaler.transform(final_data)

    # Prediction
    prediction = model.predict(scaled_data)

    st.success(f"Estimated Insurance Expense: ₹ {prediction[0]:,.2f}")