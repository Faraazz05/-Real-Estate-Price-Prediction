# 🏡 Real Estate Price Prediction — ML Regression Project

A focused **machine learning regression project** for predicting real estate prices using a **multiple linear regression model**, backed by a **Streamlit web application** with a **three-page workflow**.

This project covers the full mini-ML lifecycle: data generation, exploration, model training, evaluation, and interactive prediction — packaged cleanly for portfolio and professional use.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📂 Project Structure

```bash

real-estate-regression/
│
├── data/
│   ├── dataset.csv              # Synthetic housing dataset
│   └── generate_data.py         # Dataset generation script
│
├── notebooks/
│   └── exploration.ipynb        # EDA + baseline regression analysis
│
├── src/
│   ├── **init**.py
│   ├── data_loader.py           # Data loading and train-test split
│   ├── model.py                 # Multiple linear regression model
│   └── utils.py                 # Metrics and helper utilities
│
├── app.py                       # Streamlit application (3 pages)
├── requirements.txt             # Project dependencies
├── .gitignore                   # Ignored files
└── README.md                    # Documentation

````

---

## 🚀 Getting Started

### 1. Clone the repository and install dependencies

```bash
git clone https://github.com/yourusername/real-estate-regression.git
cd real-estate-regression
pip install -r requirements.txt
````

### 2. Generate the dataset

```bash
python data/generate_data.py
```

### 3. Explore the data and model

```bash
jupyter notebook notebooks/exploration.ipynb
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

---

## 📊 Dataset Overview

The dataset is **synthetically generated** to simulate realistic housing price behavior.

**Features include:**

* House size (square feet)
* Number of bedrooms
* Number of bathrooms
* Distance from city center
* Age of the house

**Target variable:**

* House price

Noise is intentionally added to reflect real-world variability.

---

## 🧠 Machine Learning Model

* Algorithm: **Multiple Linear Regression**
* Type: **Supervised regression**
* Train/Test split used for evaluation

**Evaluation metrics:**

* RMSE (Root Mean Squared Error)
* R² Score

Additional analysis includes:

* Regression coefficients
* Residual distribution
* Feature influence interpretation

---

## 🖥️ Streamlit Application (3 Pages)

The Streamlit app provides an interactive interface divided into three logical pages:

1. **Dataset Explorer**

   * View raw data
   * Summary statistics
   * Feature inspection

2. **Model Training & Evaluation**

   * Train regression model
   * Display metrics (RMSE, R²)
   * View coefficients and residuals

3. **Price Prediction**

   * Input house features using sliders
   * Get real-time predicted price output

---

## 🧾 Authorship

Forged with intent.

```bash
# 𓋹 Faraz
__fz_anchor__ = (
    1755693780,
    "time > memory"
)
```

---

## 📌 Notes

* The dataset is **fully synthetic** and generated programmatically.
* The project is intended to demonstrate:

  * Regression modeling
  * ML workflow clarity
  * Streamlit-based ML app deployment
* Designed as a **mini but complete ML project**, suitable for learning.
