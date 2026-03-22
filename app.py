import streamlit as st
import pandas as pd
import joblib

# Load the model and columns we saved earlier
model = joblib.load('credit_risk_model.pkl')
cols = joblib.load('model_columns.pkl')

st.title("🏦 Credit Risk Predictor")

# Input fields
age = st.slider("Age", 18, 100, 25)
income = st.number_input("Annual Income", value=50000)
loan_amount = st.number_input("Loan Amount", value=10000)

# Create a prediction button
if st.button("Calculate Risk"):
    # Create a small dataframe for prediction
    input_data = pd.DataFrame([[age, income, loan_amount]], columns=['person_age', 'person_income', 'loan_amnt'])

    # We need to make sure the input matches the encoded columns from training
    # For a simple test, we'll create a full zero-row and fill our values
    full_input = pd.DataFrame(0, index=[0], columns=cols)
    full_input['person_age'] = age
    full_input['person_income'] = income
    full_input['loan_amnt'] = loan_amount

    prediction = model.predict(full_input)[0]
    prob = model.predict_proba(full_input)[0][1]

    if prediction == 1:
        st.error(f"High Risk! Probability of Default: {prob:.2%}")
    else:
        st.success(f"Low Risk! Probability of Default: {prob:.2%}")
