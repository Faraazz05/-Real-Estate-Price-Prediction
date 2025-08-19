import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.abspath(os.path.join("..")))


from Src.data_loader import load_data, split_features_target
from Src.model import RealEstateModel
from Src.utils import correlation_heatmap, residual_plot

# --------------------
# Page Config
# --------------------
st.set_page_config(
    page_title="🏡 Real Estate Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# --------------------
# Custom CSS Styling
# --------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        font-family: 'Segoe UI', sans-serif;
    }
    .card {
        padding: 1.5rem;
        background: white;
        border-radius: 12px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .metric-box {
        background: #eef2ff;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 1rem;
    }
    .success-box {
        background: #e6f9f0;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #27ae60;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------
# Load Dataset
# --------------------
@st.cache_data
def get_data():
    return load_data()

df = get_data()

# Sidebar navigation
st.sidebar.title("🔎 Navigation")
page = st.sidebar.radio("Go to", ["Explore Data", "Train Model", "Predict Price"])

# --------------------
# Explore Data
# --------------------
if page == "Explore Data":
    st.title("🏡 Real Estate Data Explorer")

    with st.container():
        st.markdown("### 📋 Dataset Preview")
        st.dataframe(df.head())

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📊 Basic Statistics")
        st.dataframe(df.describe())
    with col2:
        st.markdown("### 🔗 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(6, 5))
        correlation_heatmap(df)
        st.pyplot(fig)

# --------------------
# Train Model
# --------------------
elif page == "Train Model":
    st.title("📈 Train & Evaluate Model")

    X, y = split_features_target(df)
    model = RealEstateModel()
    metrics = model.train(X, y)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📏 Model Performance")
        st.markdown(
            f"""
            <div class="metric-box">
                <h3>RMSE</h3>
                <p><b>{metrics['rmse']:.2f}</b></p>
            </div>
            <div class="metric-box">
                <h3>R²</h3>
                <p><b>{metrics['r2']:.2f}</b></p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown("### ⚖️ Coefficients")
        coef_df = model.get_coefficients(X.columns)
        st.dataframe(coef_df)

    st.markdown("### 🟣 Residual Plot")
    y_pred = model.predict(X)
    fig, ax = plt.subplots(figsize=(7, 5))
    residual_plot(y, y_pred)
    st.pyplot(fig)

# --------------------
# Predict Price
# --------------------
elif page == "Predict Price":
    st.title("🔮 Predict House Price")

    X, y = split_features_target(df)
    model = RealEstateModel()
    model.train(X, y)

    st.markdown("### Enter House Features:")

    col1, col2 = st.columns(2)
    with col1:
        square_feet = st.slider("Square Feet", 500, 3500, 1200, 50)
        num_bedrooms = st.slider("Number of Bedrooms", 1, 5, 3)
        num_bathrooms = st.slider("Number of Bathrooms", 1, 3, 2)
    with col2:
        dist_to_city_center = st.slider("Distance to City Center (km)", 1, 30, 10)
        age_of_house = st.slider("Age of House (years)", 0, 50, 10)

    input_data = pd.DataFrame({
        "square_feet": [square_feet],
        "num_bedrooms": [num_bedrooms],
        "num_bathrooms": [num_bathrooms],
        "dist_to_city_center": [dist_to_city_center],
        "age_of_house": [age_of_house],
    })

    if st.button("✨ Predict Price"):
        price_pred = model.predict(input_data)[0]
        st.markdown(
            f"""
            <div class="success-box">
                <h2>💰 Estimated Price: ₹{price_pred:,.0f}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
