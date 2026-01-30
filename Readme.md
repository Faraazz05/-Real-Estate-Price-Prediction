# 🏡 Real Estate Price Prediction — ML Regression Project

A focused **machine learning regression project** for predicting real estate prices using a **multiple linear regression model**, backed by a **Streamlit web application** with a **three-page workflow**.

This project covers the full mini-ML lifecycle: data generation, exploration, model training, evaluation, and interactive prediction — packaged cleanly for portfolio and professional use.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📂 Project Structure

<<<<<<< HEAD
``` bash

real-estate-regression/
=======
## real-estate-regression/
```Bash 
>>>>>>> 1c580ab52cac8003634ecf9c78a40399d7bd4f5b
│
├── corpus/
│   ├── raw_prices.csv              # Synthetic housing dataset
│   └── synth_forge.py         # Dataset generation script
│
├── lab/
│   └── ground_truth.ipynb        # EDA + baseline regression analysis
│
├── forge/
│   ├── **__init__**.py
│   ├── measures.py           # Data loading and train-test split
│   ├── regressor.py                 # Multiple linear regression model
│   └── ingest.py                 # Metrics and helper utilities
│
├── app.py                       # Streamlit application (3 pages)
├── requirements.txt             # Project dependencies
├── .gitignore                   # Ignored files
└── README.md                    # Documentation

````
<<<<<<< HEAD

---
=======
-----
>>>>>>> 1c580ab52cac8003634ecf9c78a40399d7bd4f5b

## 🚀 Getting Started

### 1. Clone the repository and install dependencies

```bash
<<<<<<< HEAD
git clone https://github.com/Faraazz05/real-estate-regression.git
=======
git clone https://github.com/yourusername/Real-Estate-Price-Prediction.git
>>>>>>> 1c580ab52cac8003634ecf9c78a40399d7bd4f5b
cd real-estate-regression
pip install -r requirements.txt
````

### 2. Generate the dataset

```bash
python corpus/synth_forge.py
```

### 3. Explore the data and model

```bash
jupyter notebook lab/ground_truth.ipynb
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```
<<<<<<< HEAD

---
=======
## 📊 Features
>>>>>>> 1c580ab52cac8003634ecf9c78a40399d7bd4f5b

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

<<<<<<< HEAD
* Regression coefficients
* Residual distribution
* Feature influence interpretation

---

## 🖥️ Streamlit Application (3 Pages)
=======
## Multiple Regression ML | Program Walkthrough (Video)
[streamlit-app-2025-08-19-19-08-42.webm](https://github.com/user-attachments/assets/f7079202-1611-4215-80d9-34feed7d407f)

## 📌 Notes
>>>>>>> 1c580ab52cac8003634ecf9c78a40399d7bd4f5b

The Streamlit app provides an interactive interface divided into three logical pages:

<<<<<<< HEAD
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
* Designed as a **mini but complete ML project**, suitable for learning and portfolio use.

---

## 📄 License

This project is licensed under the **MIT License**.
=======
This project is for educational purposes to demonstrate regression and interactive ML apps.


>>>>>>> 1c580ab52cac8003634ecf9c78a40399d7bd4f5b
