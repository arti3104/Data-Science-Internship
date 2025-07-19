import streamlit as st
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier

st.title("💼 Remote Work Productivity Predictor")
st.write("Estimate productivity levels of remote employees based on behavioral and environmental factors.")

# Input Features
hours_worked = st.slider("Average Daily Work Hours", 0, 12, 6)
distractions = st.slider("Daily Distraction Time (in minutes)", 0, 240, 60)
meetings = st.slider("Average Meetings per Day", 0, 10, 2)
workspace = st.selectbox("Dedicated Workspace?", ["Yes", "No"])
exercise = st.selectbox("Exercises Regularly?", ["Yes", "No"])
flexibility = st.selectbox("Flexible Working Hours?", ["Yes", "No"])

# Encoding
binary_map = {"Yes": 1, "No": 0}
X_input = np.array([
    hours_worked,
    distractions,
    meetings,
    binary_map[workspace],
    binary_map[exercise],
    binary_map[flexibility]
]).reshape(1, -1)

# Simulated dataset
np.random.seed(42)
X_train = np.random.randint(0, 12, (300, 6))
y_train = (X_train[:, 0] - X_train[:, 1]/60 + X_train[:, 3] + X_train[:, 4]) > 7  # pseudo logic for productivity

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prediction
prediction = model.predict(X_input)[0]
confidence = model.predict_proba(X_input)[0][1]

# Result Display
st.subheader("📈 Prediction:")
if prediction:
    st.success("✅ This employee is likely to be **Highly Productive**!")
else:
    st.warning("⚠️ This employee may struggle with productivity.")

st.write(f"Confidence Score: **{round(confidence * 100, 2)}%**")

# Visualization
st.subheader("🔍 Prediction Probability")
fig = px.bar(
    x=["Low Productivity", "High Productivity"],
    y=model.predict_proba(X_input)[0],
    color=["Low Productivity", "High Productivity"],
    labels={"x": "Productivity Level", "y": "Probability"},
    color_discrete_sequence=["orange", "green"]
)
st.plotly_chart(fig)
