import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------------------------------
# 🎯 App Configuration
# ----------------------------------------------------
st.set_page_config(
    page_title="🎓 Student Result Predictor",
    page_icon="🎯",
    layout="centered"
)

# ----------------------------------------------------
# 🌈 Clean Background + Styling
# ----------------------------------------------------
st.markdown("""
<style>
/* Simple animated gradient background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(-45deg, #8E2DE2, #4A00E0, #00C9FF, #92FE9D);
    background-size: 400% 400%;
    animation: gradientBG 12s ease infinite;
}
@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Remove glass box effect */
.block-container {
    max-width: 850px;
    margin: auto;
    background: transparent !important;
    padding: 20px 30px;
}

/* Titles and text */
h1, h2, h3, p, label {
    color: #fff !important;
    text-align: center;
    font-family: 'Poppins', sans-serif;
}
hr {
    border: 1px solid rgba(255, 255, 255, 0.3);
    margin: 10px 0 25px 0;
}

/* Input styling — white box + black text */
input, select, textarea {
    background-color: #fff !important;
    color: #000 !important;
    border-radius: 6px !important;
    border: 1px solid #ccc !important;
}

/* Predict button */
.stButton > button {
    background: linear-gradient(90deg, #4A00E0, #8E2DE2);
    color: white !important;
    border: none;
    border-radius: 8px;
    padding: 0.6em 2em;
    font-size: 16px;
    font-weight: 600;
    transition: all 0.3s ease;
    display: block;
    margin: 0 auto;
}
.stButton > button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #8E2DE2, #4A00E0);
}

/* Alert boxes */
div.stAlert {
    background-color: rgba(255,255,255,0.25) !important;
    color: #fff !important;
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.35);
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 🏫 Title
# ----------------------------------------------------
st.markdown("""
<h1>🎓 Student Result Prediction App</h1>
<p>Predict whether a student will <b>PASS</b> or <b>FAIL</b> based on academic and behavioral data.</p>
<hr>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 📦 Load Model & Scaler
# ----------------------------------------------------
try:
    model = joblib.load("logistic_model.joblib")
    scaler = joblib.load("scaler.joblib")
except Exception as e:
    st.error(f"❌ Model or Scaler not loaded.\n\nError: {e}")
    st.stop()

# ----------------------------------------------------
# 🧾 User Inputs
# ----------------------------------------------------
st.subheader("🧍 Enter Student Details")

with st.form("student_form"):
    col1, col2 = st.columns(2)

    with col1:
        Gender = st.selectbox("Gender", ["Male", "Female"])
        Age = st.number_input("Age", min_value=15, max_value=22, step=1)
        Parent_Education_Level = st.selectbox(
            "Parent Education Level",
            ["Primary", "Secondary", "High School", "Bachelor", "Master", "PhD"]
        )
        Study_Time = st.number_input("Study Time (hours/day)", min_value=1, max_value=10, step=1)
        Failures = st.number_input("Number of Failures", min_value=0, max_value=5, step=1)
        Absences = st.number_input("Number of Absences", min_value=0, max_value=50, step=1)

    with col2:
        Tutoring = st.selectbox("Attends Tutoring?", ["Yes", "No"])
        Test_Preparation_Course = st.selectbox("Test Preparation Course?", ["Yes", "No"])
        Previous_Score = st.number_input("Previous Score (%)", min_value=0, max_value=100, step=1)
        Motivation_Level = st.number_input("Motivation Level (1–10)", min_value=1, max_value=10, value=1)
        Stress_Level = st.number_input("Stress Level (1–10)", min_value=1, max_value=10, value=1)
        Final_Grade = st.number_input("Final Grade (out of 20)", min_value=0, max_value=20, step=1)

    # Center predict button
    col_center = st.columns([1, 1, 1])
    with col_center[1]:
        submitted = st.form_submit_button("🔮 Predict Result", use_container_width=False)

# ----------------------------------------------------
# 🧠 Prediction Logic
# ----------------------------------------------------
if submitted:
    input_data = pd.DataFrame({
        'Gender': [Gender],
        'Age': [Age],
        'Parent_Education_Level': [Parent_Education_Level],
        'Study_Time': [Study_Time],
        'Failures': [Failures],
        'Absences': [Absences],
        'Tutoring': [Tutoring],
        'Test_Preparation_Course': [Test_Preparation_Course],
        'Previous_Score': [Previous_Score],
        'Motivation_Level': [Motivation_Level],
        'Stress_Level': [Stress_Level],
        'Final_Grade': [Final_Grade]
    })

    input_encoded = input_data.replace({
        'Gender': {'Male': 0, 'Female': 1},
        'Parent_Education_Level': {
            'Primary': 0, 'Secondary': 1, 'High School': 2,
            'Bachelor': 3, 'Master': 4, 'PhD': 5
        },
        'Tutoring': {'Yes': 1, 'No': 0},
        'Test_Preparation_Course': {'Yes': 1, 'No': 0}
    })

    try:
        input_scaled = scaler.transform(input_encoded)
    except Exception as e:
        st.error(f"⚠️ Input scaling error: {e}")
        st.stop()

    prediction = model.predict(input_scaled)[0]
    result = "PASS" if prediction == 1 else "FAIL"

    st.markdown("<hr>", unsafe_allow_html=True)

    if result == "PASS":
        st.success("🎉 **Prediction: PASS** — The student is likely to pass! 🌟")
        st.balloons()
    else:
        st.error("⚠️ **Prediction: FAIL** — The student might fail. Needs improvement.")