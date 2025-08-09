import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Must be first Streamlit command
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

# Load model and encoders
model = joblib.load('heart_disease_model.pkl')
encoders = joblib.load('encoders.pkl')

# Main CSS styling including button fix
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        
        * {
            box-sizing: border-box;
        }
        
        html, body, .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #ffffff;
            font-family: 'Inter', sans-serif;
        }
        
        .block-container {
            padding-top: 1rem !important;
            max-width: 1200px !important;
        }
        
        .main-title {
            color: #FFFFFF !important;
            font-size: 3rem !important;
            font-weight: 800 !important;
            text-align: center !important;
            margin-bottom: 1rem !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5) !important;
        }
        
        .subtitle {
            color: #E5E7EB !important;
            font-size: 1.2rem !important;
            text-align: center !important;
            margin-bottom: 2rem !important;
        }
        
        .section-header {
            color: #FFFFFF !important;
            font-size: 1.5rem !important;
            font-weight: 700 !important;
            margin: 2rem 0 1rem 0 !important;
        }
        
        .stForm {
            background: rgba(255,255,255,0.1) !important;
            border-radius: 15px !important;
            padding: 2rem !important;
            backdrop-filter: blur(10px) !important;
            border: 1px solid rgba(255,255,255,0.2) !important;
        }
        
        .stNumberInput label,
        .stSelectbox label,
        .stSlider label {
            color: #FFFFFF !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
        }
        
        .stNumberInput input {
            background-color: #FFFFFF !important;
            color: #000000 !important;
            border: 2px solid #4F46E5 !important;
            border-radius: 8px !important;
            padding: 12px !important;
            font-size: 1rem !important;
            height: 50px !important;
        }
        
        .stNumberInput input:focus {
            border-color: #7C3AED !important;
            box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.1) !important;
        }
        
        .stSelectbox > div > div {
            background-color: #FFFFFF !important;
            border: 2px solid #4F46E5 !important;
            border-radius: 8px !important;
            min-height: 50px !important;
        }
        
        .stSelectbox > div > div > div {
            color: #000000 !important;
            padding: 12px !important;
            font-size: 1rem !important;
        }
        
        .stSlider > div > div > div > div {
            background: #4F46E5 !important;
        }
        
        .stSlider > div > div > div > div > div {
            background: #FFFFFF !important;
            border: 3px solid #4F46E5 !important;
        }
        
        /* Submit button style */
        div[data-testid="stForm"] button[kind="primary"] {
            background-color: #4F46E5 !important;   /* Dark purple */
            color: #FFFFFF !important;              /* White text */
            font-weight: 800 !important;
            font-size: 1.2rem !important;
            padding: 15px 30px !important;
            border-radius: 10px !important;
            width: 100% !important;
            margin-top: 1.5rem !important;
            cursor: pointer !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
            border: none !important;
            text-shadow: none !important;
            transition: all 0.3s ease !important;
        }
        div[data-testid="stForm"] button[kind="primary"] * {
            color: #FFFFFF !important;
        }
        div[data-testid="stForm"] button[kind="primary"]:hover {
            background: linear-gradient(135deg, #7C3AED, #A855F7) !important;
            color: #FFFFFF !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 5px 15px rgba(0,0,0,0.3) !important;
        }
        
        /* Results container and other styles (unchanged, but included for completeness) */
        .results-section {
            margin-top: 2rem !important;
            padding: 2.5rem !important;
            border-radius: 20px !important;
            backdrop-filter: blur(15px) !important;
            border: 1px solid rgba(255, 255, 255, 0.25) !important;
            box-shadow: 0 8px 24px rgba(0,0,0,0.3) !important;
            max-width: 700px !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }
        .results-section h2 {
            font-size: 2.2rem !important;
            font-weight: 900 !important;
            margin-bottom: 1rem !important;
            color: #FFE600 !important;
            text-align: center !important;
            text-shadow: 1px 1px 3px rgba(0,0,0,0.6);
        }
        .stSuccess {
            background: rgba(34, 197, 94, 0.15) !important;
            border: 3px solid #22C55E !important;
            color: #DFFFE3 !important;
            font-weight: 700 !important;
            font-size: 1.4rem !important;
            border-radius: 15px !important;
            padding: 1.5rem 2rem !important;
            text-align: center !important;
            box-shadow: 0 4px 15px rgba(34, 197, 94, 0.5);
        }
        .stError {
            background: rgba(239, 68, 68, 0.15) !important;
            border: 3px solid #EF4444 !important;
            color: #FFE6E6 !important;
            font-weight: 700 !important;
            font-size: 1.4rem !important;
            border-radius: 15px !important;
            padding: 1.5rem 2rem !important;
            text-align: center !important;
            box-shadow: 0 4px 15px rgba(239, 68, 68, 0.5);
        }
        .stInfo {
            background: rgba(59, 130, 246, 0.15) !important;
            border: 3px solid #3B82F6 !important;
            color: #DDE9FF !important;
            font-weight: 600 !important;
            font-size: 1.25rem !important;
            border-radius: 15px !important;
            padding: 1.2rem 2rem !important;
            margin-top: 1rem !important;
            text-align: center !important;
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
        }
        .confidence-text {
            margin-top: 1.5rem !important;
            background: rgba(255, 255, 255, 0.15) !important;
            color: #FFFFF0 !important;
            font-weight: 400 !important;
            font-size: 1rem !important;
            padding: 0.2rem !important;
            border-radius: 12px !important;
            text-align: center !important;
            width: fit-content !important;
            box-shadow: 0 2px 10px rgba(255,255,255,0.3);
            text-shadow: 0 0 5px rgba(255,255,255,0.6);
            letter-spacing: 0.05em;
        }
        /* Hide Streamlit default menu, footer, header */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
    </style>
""", unsafe_allow_html=True)

# App UI
st.markdown('<h1 class="main-title">💓 Heart Disease Risk Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Cardiovascular risk assessment</p>', unsafe_allow_html=True)
st.markdown('<h2 class="section-header">🩺 Patient Information</h2>', unsafe_allow_html=True)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("👤 Age (years)", min_value=20, max_value=100, value=50)
        sex = st.selectbox("⚥ Sex", options=encoders['Sex'].classes_)
        chest_pain = st.selectbox("💔 Chest Pain Type", options=encoders['ChestPainType'].classes_)
        fasting_bs = st.selectbox("🩸 Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        rest_ecg = st.selectbox("📈 Resting ECG", options=encoders['RestingECG'].classes_)

    with col2:
        rest_bp = st.number_input("🫀 Resting Blood Pressure (mmHg)", min_value=80, max_value=200, value=120)
        cholesterol = st.number_input("🧪 Cholesterol (mg/dl)", min_value=100, max_value=400, value=200)
        max_hr = st.number_input("💓 Maximum Heart Rate", min_value=60, max_value=210, value=150)
        exercise_angina = st.selectbox("🏃 Exercise Induced Angina", options=encoders['ExerciseAngina'].classes_)
        st_slope = st.selectbox("📊 ST Slope", options=encoders['ST_Slope'].classes_)
    
    oldpeak = st.slider("📉 Oldpeak (ST depression)", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
    
    submitted = st.form_submit_button("🔍 ANALYZE HEART DISEASE RISK")


if submitted:
    try:
        input_data = pd.DataFrame({
            'Age': [age],
            'RestingBP': [rest_bp],
            'Cholesterol': [cholesterol],
            'FastingBS': [fasting_bs],
            'MaxHR': [max_hr],
            'Oldpeak': [oldpeak],
            'Sex': [encoders['Sex'].transform([sex])[0]],
            'ChestPainType': [encoders['ChestPainType'].transform([chest_pain])[0]],
            'RestingECG': [encoders['RestingECG'].transform([rest_ecg])[0]],
            'ExerciseAngina': [encoders['ExerciseAngina'].transform([exercise_angina])[0]],
            'ST_Slope': [encoders['ST_Slope'].transform([st_slope])[0]]
        })

        prediction = model.predict(input_data)[0]
        confidence = model.predict_proba(input_data)[0][prediction]

        st.markdown('<h2>📋 Risk Assessment Results</h2>', unsafe_allow_html=True)

        if prediction == 1:
            st.markdown('<div class="stError">🔴 HIGH RISK OF HEART DISEASE DETECTED</div>', unsafe_allow_html=True)
            st.markdown(f'<p class="confidence-text">🎯 Confidence Level: {round(confidence * 100, 2)}%</p>', unsafe_allow_html=True)
            st.markdown('<div class="stInfo">💡 RECOMMENDATION: Please consult with a cardiologist immediately for comprehensive evaluation.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="stSuccess">🟢 LOW RISK OF HEART DISEASE</div>', unsafe_allow_html=True)
            st.markdown(f'<p class="confidence-text">🎯 Confidence Level: {round(confidence * 100, 2)}%</p>', unsafe_allow_html=True)
            st.markdown('<div class="stInfo">💚 RECOMMENDATION: Continue maintaining a healthy lifestyle with regular check-ups.</div>', unsafe_allow_html=True)
            st.balloons()

    except Exception as e:
        st.error(f"⚠️ ERROR: {str(e)}")
