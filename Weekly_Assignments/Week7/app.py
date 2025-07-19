import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LogisticRegression

# Title
st.title("❤️ Heart Disease Risk Predictor")
st.write("This app predicts the risk of heart disease based on health indicators.")

# Sidebar Inputs
age = st.slider("Age", 20, 80, 45)
resting_bp = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.slider("Cholesterol Level (mg/dL)", 100, 400, 220)
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL?", ["No", "Yes"])
chest_pain = st.selectbox("Chest Pain Type", ["Typical", "Atypical", "Non-anginal", "Asymptomatic"])

# Convert categorical inputs
fbs_map = {"No": 0, "Yes": 1}
cp_map = {"Typical": 0, "Atypical": 1, "Non-anginal": 2, "Asymptomatic": 3}

X_input = np.array([
    age,
    resting_bp,
    cholesterol,
    max_hr,
    fbs_map[fasting_bs],
    cp_map[chest_pain]
]).reshape(1, -1)

# Simulate training data
X_train = np.random.randint(0, 200, (150, 6))
y_train = (X_train[:, 0] + X_train[:, 1] + X_train[:, 2]) > 300  # Simulated risk rule

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
prediction = model.predict(X_input)[0]
proba = model.predict_proba(X_input)[0][1]

# Output
st.subheader("📊 Prediction:")
if prediction == 1:
    st.error("⚠️ At **High Risk** of Heart Disease")
else:
    st.success("✅ **Low Risk** of Heart Disease")

st.write(f"Prediction Confidence: **{round(proba * 100, 2)}%**")

# Visualization
st.subheader("📈 Confidence Score Chart")
fig = px.bar(
    x=["Low Risk", "High Risk"],
    y=model.predict_proba(X_input)[0],
    labels={"x": "Outcome", "y": "Probability"},
    color=["Low Risk", "High Risk"],
    color_discrete_sequence=["green", "red"]
)
st.plotly_chart(fig)
