import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import os, streamlit as st

# Page config
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

# Load model and encoder columns
model = joblib.load("Project/churn_model.pkl")
encoder = joblib.load("Project/encoder.pkl")
df = pd.read_csv("Project/Dataset/Telco-Customer-Churn.csv")

# Sidebar navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", ["🔮 Prediction", "📈 Dashboard"])

# -------------------------------------
# 🔮 Prediction Page
# -------------------------------------
if page == "🔮 Prediction":
    st.title("Customer Churn Prediction")

    st.markdown("Fill the form to predict if a customer is likely to churn.")

    # Input form
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 10)
    monthly = st.slider("Monthly Charges", 20, 150, 70)
    total = st.slider("Total Charges", 20, 10000, 1000)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

    input_data = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [1 if senior == "Yes" else 0],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "MonthlyCharges": [monthly],
    "TotalCharges": [total],
    "Contract": [contract]
    })

    encoder_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                'tenure', 'MonthlyCharges', 'TotalCharges', 'Contract']

    # Reorder columns to match training
    X = encoder.transform(input_data)

    # Predict
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1] * 100

    st.subheader("📢 Result")
    st.metric("Churn Probability", f"{proba:.2f}%")
    st.write("Prediction:", "✅ Customer will churn" if prediction == 1 else "🟢 Customer will stay")

    # Retention suggestion
    st.subheader("💡 Retention Suggestion")
    if proba > 75:
        st.warning("⚠️ High risk: Suggest 15% discount or call from retention team.")
    elif proba > 50:
        st.info("Moderate risk: Recommend email campaign or service upgrade.")
    else:
        st.success("Low risk: No action needed.")

    # SHAP Explainability
    st.subheader("🧠 Feature Importance")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        shap.summary_plot(shap_values, X)
        st.pyplot(bbox_inches='tight')
    except Exception as e:
        st.warning("SHAP explainability failed or not supported for this model.")
        st.text(str(e))

# -------------------------------------
# 📈 Dashboard Page
# -------------------------------------
elif page == "📈 Dashboard":
    st.title("Churn Analytics Dashboard")

    # Load dataset
    df = pd.read_csv("Project/Dataset/Telco-Customer-Churn.csv")

    # Show churn distribution
    st.subheader("1️⃣ Churn Distribution")
    st.bar_chart(df['Churn'].value_counts())

    # Churn by Contract
    st.subheader("2️⃣ Churn by Contract")
    fig1 = px.histogram(df, x="Contract", color="Churn", barmode="group", title="Churn by Contract Type")
    st.plotly_chart(fig1)

    # Churn by Senior Citizen
    st.subheader("3️⃣ Churn by Senior Citizen")
    fig2 = px.histogram(df, x="SeniorCitizen", color="Churn", barmode="group", title="Churn vs Senior Citizen")
    st.plotly_chart(fig2)

    # Churn by Gender
    st.subheader("4️⃣ Churn by Gender")
    fig3 = px.histogram(df, x="gender", color="Churn", barmode="group", title="Churn by Gender")
    st.plotly_chart(fig3)

    #st.info("Add more plots like Churn by Tenure, Payment Method, Internet Service, etc.")
