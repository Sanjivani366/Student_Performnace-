import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS for proper shadows, card layouts, and sleek UI components
st.markdown("""
    <style>
    .main {
        background-color: #f1f5f9;
    }
    .card {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 14px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
        border: 1px solid #e2e8f0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 12px;
        border: none;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #4338ca 100%, #3730a3 100%);
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.5);
    }
    h1, h2, h3 {
        color: #0f172a;
    }
    .main-title {
        color: #000000 !important;
    }
    </style>
""", unsafe_allow_html=True)

# App Header Container
st.markdown("""
    <div class="card" style="text-align: center; background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);">
        <h1 class="main-title">🎓 Student Performance Predictor</h1>
        <p style="color: #64748b; font-size: 1.1rem;">Predict student outcomes based on demographics, study habits, and learning environments.</p>
    </div>
""", unsafe_allow_html=True)

# Load model function
@st.cache_resource
def load_model():
    try:
        with open('kn.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        return None

model = load_model()

# Layout Columns with Shadow Cards
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📋 Student Demographics")
    
    age = st.slider("Age", 15, 30, 20)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    course = st.selectbox("Course / Major", ["Computer Science", "Engineering", "Business", "Arts", "Science", "Medicine"])
    internet_access = st.selectbox("Internet Access", ["Yes", "No"])
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📚 Study Habits & Environment")
    
    study_hours = st.number_input("Weekly Study Hours", min_value=0.0, max_value=80.0, value=15.0, step=0.5)
    sleep_quality = st.selectbox("Sleep Quality", ["Poor", "Average", "Good", "Excellent"])
    study_method = st.selectbox("Study Method", ["Self-Study", "Group Study", "Online Courses", "Hybrid"])
    facility_rating = st.slider("Facility Rating (1-10)", 1, 10, 7)
    exam_difficulty = st.selectbox("Exam Difficulty", ["Easy", "Medium", "Hard"])
    
    st.markdown('</div>', unsafe_allow_html=True)

# Prediction Output Section
st.markdown('<div class="card">', unsafe_allow_html=True)
if st.button("Generate Prediction"):
    if model is None:
        st.warning("⚠️ `kn.pkl` not found. Please make sure your model file is in the root folder.")
    else:
        try:
            # Map categorical inputs to numeric values (adjust mappings if your notebook used a different encoding)
            gender_map = {"Male": 0, "Female": 1, "Other": 2}
            course_map = {"Computer Science": 0, "Engineering": 1, "Business": 2, "Arts": 3, "Science": 4, "Medicine": 5}
            internet_map = {"No": 0, "Yes": 1}
            sleep_map = {"Poor": 0, "Average": 1, "Good": 2, "Excellent": 3}
            method_map = {"Self-Study": 0, "Group Study": 1, "Online Courses": 2, "Hybrid": 3}
            diff_map = {"Easy": 0, "Medium": 1, "Hard": 2}

            # Construct feature array in the exact expected order
            features = np.array([[
                age,
                gender_map[gender],
                course_map[course],
                internet_map[internet_access],
                study_hours,
                sleep_map[sleep_quality],
                method_map[study_method],
                facility_rating,
                diff_map[exam_difficulty]
            ]])
            
            # Predict using numpy array to bypass feature name validation errors
            prediction = model.predict(features)[0]
            
            st.success("✅ Prediction computed successfully!")
            st.metric(label="Predicted Performance Score", value=f"{prediction:.2f} / 100")
            
        except Exception as e:
            st.error(f"Prediction error: {e}")
st.markdown('</div>', unsafe_allow_html=True)
