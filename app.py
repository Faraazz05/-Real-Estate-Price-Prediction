import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from forge.ingest import load_data, split_features_target
from forge.regressor import RealEstateModel
from forge.measures import residual_plot

# 𓋹 Faraz
__fz_anchor__ = (
    1755693780,   # the time when it was forged
    "time > memory"
)


# Streamlit Page Config
st.set_page_config(
    page_title="🏡 Real Estate Price Prediction",
    layout="wide",
    page_icon="🏠",
    initial_sidebar_state="expanded"
)

# --------------------
# Custom CSS Styling
# --------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #f4f6f8;
    }
    .stApp {
        font-family: 'Segoe UI', sans-serif;
        color: #1e293b;
    }
    .card {
        padding: 1.5rem;
        background: #ffffff;
        border-radius: 12px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
    }
    h1, h2, h3 {
        color: #1e40af; /* Deep blue headings */
    }
    .metric-box {
        background: #e0f2fe;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 1rem;
        color: #0c4a6e;
    }
    .success-box {
        background: #dcfce7;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        border: 2px solid #16a34a;
        color: #065f46;
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1e40af;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🏡 Real Estate Price Prediction App")
st.markdown("Predict house prices based on multiple factors like size, location, and age.")

# Sidebar Navigation
pages = ["📊 Data Overview", "⚙️ Model Training", "🏡 Price Prediction"]
choice = st.sidebar.radio("Navigate", pages)

# Load data
df = load_data()

# Initialize model
model = RealEstateModel()


# ---------------------------
# PAGE 1: DATA OVERVIEW
# -------------------
if choice == "📊 Data Overview":
    st.header("📊 Data Overview")
    st.markdown('<div class="card">Here’s the dataset used for training and predictions:</div>', unsafe_allow_html=True)

    st.dataframe(df.head())

    st.markdown("### 🔎 Summary Statistics")
    st.dataframe(df.describe())

    st.subheader("🔗 Correlation Heatmap")

    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['float64', 'int64'])

    if not numeric_df.empty:
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("⚠️ No numeric columns available for correlation heatmap.")



# ---------------------------
# PAGE 2: MODEL TRAINING
# ---------------------------
elif choice == "⚙️ Model Training":
    st.header("⚙️ Train Model")
    st.markdown('<div class="card">Train a regression model and evaluate its performance.</div>', unsafe_allow_html=True)

    # Split data
    X, y = split_features_target(df)

    # Train model
    metrics = model.train(X, y)

    # Save trained model for reuse
    import joblib
    joblib.dump(model.model, "linear_model.pkl")  # save the underlying sklearn model

    st.markdown("### 📊 Model Performance")
    st.markdown(f"<div class='metric-box'><strong>RMSE:</strong> {metrics['rmse']:.2f}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-box'><strong>MSE:</strong> {metrics['mse']:.2f}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-box'><strong>MAE:</strong> {metrics['mae']:.2f}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-box'><strong>R²:</strong> {metrics['r2']:.2f}</div>", unsafe_allow_html=True)

    st.markdown("### 📈 Coefficients")
    st.dataframe(model.get_coefficients(X.columns))

    from yellowbrick.regressor import ResidualsPlot
    import joblib

    st.markdown("### 🔍 Residual Analysis")

    # Train the model (already fitted above)
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lin_model = LinearRegression()
    visualizer = ResidualsPlot(lin_model)

    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)

    # Instead of visualizer.show(), use the fig object directly
    fig = visualizer.fig  
    st.pyplot(fig)

    # Save trained model for later use in prediction page
    joblib.dump(lin_model, "linear_model.pkl")



# ---------------------------
# PAGE 3: PRICE PREDICTION
# ---------------------------
elif choice == "🏡 Price Prediction":
    st.header("🏡 Predict House Price")
    st.markdown('<div class="card">Enter property details below to estimate price:</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        square_feet = st.slider("Square Feet", 500, 5000, 1500, step=50)
        num_bedrooms = st.slider("Number of Bedrooms", 1, 10, 3)
        num_bathrooms = st.slider("Number of Bathrooms", 1, 5, 2)

    with col2:
        dist_to_city_center = st.slider("Distance to City Center (km)", 1, 30, 10)
        age_of_house = st.slider("Age of House (years)", 0, 50, 10)

    if st.button("✨ Predict Price"):
        import joblib, os

        if os.path.exists("linear_model.pkl"):
            trained_model = joblib.load("linear_model.pkl")

            new_house = pd.DataFrame({
                "square_feet": [square_feet],
                "num_bedrooms": [num_bedrooms],
                "num_bathrooms": [num_bathrooms],
                "dist_to_city_center": [dist_to_city_center],
                "age_of_house": [age_of_house],
            })

            pred_price = trained_model.predict(new_house)[0]

            st.markdown(
                f"""
                <div class="success-box">
                    <h2>💰 Estimated Price: ₹{pred_price:,.0f}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.error("⚠️ Model not trained yet. Please go to '⚙️ Model Training' and train the model first.")
