import streamlit as st
import pandas as pd
import joblib

model = joblib.load('model.pkl')
feature_names = joblib.load('feature_names.pkl')

st.title("Will this customer churn?")
st.write("Fill in the customer details and click Predict.")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Months as customer", 0, 72, 12)
    monthly = st.slider("Monthly charges ($)", 0, 120, 65)
    contract = st.selectbox("Contract type",
        ["Month-to-month", "One year", "Two year"])

with col2:
    internet = st.selectbox("Internet service",
        ["DSL", "Fiber optic", "No"])
    tech_support = st.selectbox("Tech support", ["Yes", "No"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])

if st.button("Predict churn risk"):
    row = pd.DataFrame([{
        'tenure': tenure,
        'MonthlyCharges': monthly,
        'TotalCharges': monthly * tenure,
        'SeniorCitizen': 0,
        'Contract_One year': int(contract == "One year"),
        'Contract_Two year': int(contract == "Two year"),
        'InternetService_Fiber optic': int(internet == "Fiber optic"),
        'InternetService_No': int(internet == "No"),
        'TechSupport_Yes': int(tech_support == "Yes"),
        'PaperlessBilling_Yes': int(paperless == "Yes"),
    }]).reindex(columns=feature_names, fill_value=0)

    probability = model.predict_proba(row)[0][1]

    st.markdown("---")
    st.metric("Churn probability", f"{probability:.0%}")

    if probability > 0.6:
        st.error("High risk — this customer is likely to leave!")
    elif probability > 0.3:
        st.warning("Medium risk — worth keeping an eye on.")
    else:
        st.success("Low risk — this customer is likely staying.")