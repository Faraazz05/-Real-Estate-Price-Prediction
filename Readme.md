# 🏡 Real Estate Price Prediction with Multiple Regression

A mini-project that demonstrates **multiple linear regression** on a synthetic housing dataset.  
The project includes data generation, exploratory analysis, model training, and a Streamlit web app.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
---

## 📂 Project Structure

## real-estate-regression/
│
├── data/
│ ├── dataset.csv # Generated dataset
│ └── generate_data.py # Script to generate synthetic dataset
│
├── notebooks/
│ └── exploration.ipynb # Jupyter notebook for EDA + baseline regression
│
├── src/
│ ├── init.py
│ ├── data_loader.py # Load & split data
│ ├── model.py # Regression model class
│ └── utils.py # Helper functions (plots, metrics)
│
├── app.py # Streamlit app
├── requirements.txt # Dependencies
├── .gitignore # Ignore unnecessary files
└── README.md # Project documentation


---

## 🚀 Getting Started

### 1. Clone repo & install dependencies
```bash
git clone https://github.com/yourusername/Real-Estate-Price-Prediction.git
cd real-estate-regression
pip install -r requirements.txt

2. Generate dataset
python data/generate_data.py

3. Explore in notebook
jupyter notebook notebooks/exploration.ipynb

4. Run Streamlit app
streamlit run app.py

📊 Features

Synthetic Real Estate Dataset

House size (sq ft), bedrooms, bathrooms, distance to city, age of house, price

Exploratory Data Analysis

Stats, correlation heatmap, price distribution

Model Training

Multiple linear regression

Metrics: RMSE, R²

Coefficients table

Residual analysis

Interactive Streamlit App

Dataset explorer

Model training & evaluation

House price predictor with sliders



📌 Notes

Data is synthetic, generated with a formula + noise.

This project is for educational purposes to demonstrate regression and interactive ML apps.
