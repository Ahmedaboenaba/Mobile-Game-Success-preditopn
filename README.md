# 📱 Mobile Game Success Prediction

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Library](https://img.shields.io/badge/Library-XGBoost%20|%20LightGBM%20|%20Optuna-orange)
![Task](https://img.shields.io/badge/Task-Regression-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## 📌 Project Overview
This project is a competitive machine learning solution designed to predict the **Average User Rating** of mobile games based on various metadata attributes. The goal is to build a regression model that minimizes the Root Mean Squared Error (RMSE) between predicted and actual ratings (on a scale of 0.0 to 5.0).

This solution utilizes **Advanced Feature Engineering**, **Hyperparameter Optimization (Optuna)**, and **Stacking Ensembles** to achieve high-performance results.

---
## 📌 Competition link
https://www.kaggle.com/competitions/predicting-mobile-game-success/data

## 📂 Dataset
The dataset consists of mobile game metadata with the following key features:
* **Target:** `Average User Rating` (Float 0.0 - 5.0)
* **Numerical:** `Price`, `Size` (bytes), `User Rating Count`.
* **Categorical/Text:** `Name`, `Description`, `Developer`, `Age Rating`.
* **Complex Lists:** `Languages` (ISO codes), `Genres`, `In-app Purchases`.
* **Temporal:** `Original Release Date`, `Current Version Release Date`.

---

## 🛠️ Methodology & Pipeline

### 1. Data Preprocessing & Sanitization
* **Cleaning:** Removed rows with missing target values to prevent model crashes.
* **Imputation:** Filled missing feature values (e.g., `Price` treated as 0/Free, dates treated as new) to maintain dataset integrity.
* **Sanitization:** Regex-based cleaning of column names to ensure compatibility with LightGBM/XGBoost.

### 2. Feature Engineering (The "Secret Sauce")
Raw data was transformed into meaningful numerical signals:
* **Temporal Features:** Calculated `App_Age_Days`, `Days_Since_Update`, and `Update_Lag` to capture app freshness and developer commitment.
* **Text Meta-Features:** Extracted lengths of `Description`, `Name`, and `Subtitle` (proxies for ASO effort).
* **List Parsing:** Counted supported `Languages`, `Genres`, and `In-App Purchase` items.
* **Financial Features:** Extracted `Max_IAP_Price`, `Mean_IAP_Price`, and created `Is_Free` and `Price_Log` flags.
* **Developer Impact:** Utilized **Count Encoding** on the `Developer` column to capture portfolio size and reputation.

### 3. Model Architecture
We moved beyond simple regression to a robust ensemble approach:
* **Base Models:** * **XGBoost Regressor:** Optimized for structured data.
    * **LightGBM Regressor:** Efficient handling of large datasets and categorical features.
* **Meta-Learner:** * **Stacking Regressor:** Combines predictions from base models using a **Ridge Regression** final estimator to minimize overfitting.

### 4. Optimization
* **Optuna:** Used for automated Bayesian hyperparameter tuning to find the optimal `learning_rate`, `max_depth`, and `subsample` values.

---

## 🚀 Installation & Usage

### Prerequisites
Ensure you have Python installed. Install the required dependencies:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm optuna
