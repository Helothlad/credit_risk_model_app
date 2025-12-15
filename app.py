import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load("xgb_credit_model.pkl")

# FIX 1: Changed .okl to .pkl
encoders = {col: joblib.load(f"{col}_encoder.pkl") for col in ["Sex", "Housing", "Saving accounts", "Checking account"]}

st.title("Credit Risk Prediction App")
st.write("Enter applicant information to predict if the credit risk is good or bad.")

age = st.number_input("Age", min_value=18, max_value=80, value=30)

# FIX 2: Changed selectedbox to selectbox
sex = st.selectbox("Sex", ["male", "female"])

job = st.number_input("Job (0-3)", min_value=0, max_value=3, value=1)
housing = st.selectbox("Housing", ["own", "rent", "free"])

# FIX 3: Changed "rick" to "rich" to match standard data
saving_accounts = st.selectbox("Saving Accounts", ["little", "moderate", "rich", "quite rich"])
checking_account = st.selectbox("Checking Accounts", ["little", "moderate", "rich"])

credit_amount = st.number_input("Credit Amount", min_value=0, value=1000)
duration = st.number_input("Duration (months)", min_value=1, value=12)

input_df = pd.DataFrame({
    "Age": [age],
    "Sex": [encoders["Sex"].transform([sex])[0]],
    "Job": [job],
    "Housing": [encoders["Housing"].transform([housing])[0]],
    "Saving accounts": [encoders["Saving accounts"].transform([saving_accounts])[0]],
    "Checking account": [encoders["Checking account"].transform([checking_account])[0]],
    "Credit amount": [credit_amount],
    "Duration": [duration]
})

if st.button("Predict Risk"):
    pred = model.predict(input_df)
    
    if pred == 1:
        st.success("The predicted risk is good.")
    else:
        st.error("The predicted risk is bad.")