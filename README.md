# 📱 Mobile Game Success Prediction

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Library](https://img.shields.io/badge/Library-XGBoost%20|%20LightGBM%20|%20Optuna-orange)
![Task](https://img.shields.io/badge/Task-Regression-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## 📌 Project Overview
This project is a competitive machine learning solution designed to predict the **Average User Rating** of mobile games based on various metadata attributes. The goal is to build a regression model that minimizes the Root Mean Squared Error (RMSE) between predicted and actual ratings (on a scale of 0.0 to 5.0).

This solution utilizes **Advanced Feature Engineering**, **Hyperparameter Optimization (Optuna)**, and **Stacking Ensembles** to achieve high-performance results.

---

## 📌 Kaggle Competition
https://www.kaggle.com/competitions/predicting-mobile-game-success/overview

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

```

### Project Structure

This repository contains three main scripts to replicate the workflow:

| Script Name | Description |
| --- | --- |
| **`1_tuner.py`** | Runs **Optuna** trials to find the best hyperparameters for XGBoost/LightGBM. |
| **`2_engineer.py`** | Validates the impact of new features (Feature Engineering) using Cross-Validation. |
| **`3_solver.py`** | Trains the final **Stacking Ensemble** and generates the `submission.csv` file. |

### How to Run

1. **Place Data:** Ensure `train.csv` and `test.csv` are in the root directory.
2. **Tune Parameters:**
```bash
python 1_tuner.py

```


*Copy the printed "Best Params" dictionary.*
3. **Generate Submission:**
*Paste the params into `3_solver.py`.*
```bash
python 3_solver.py

```



---

## 📊 Results

| Model / Approach | RMSE Score | Notes |
| --- | --- | --- |
| **Baseline (Voting Regressor)** | 0.23695 | Default params, basic cleaning. |
| **Tuned XGBoost** | 0.18472 | Manual tuning (LR=0.01). |
| **Stacking + Feature Eng.** | **< 0.18000** | *Target Performance* (Current SOTA ~0.166). |

---

## 👤 Author

**Ahmed Jaber** *Bioinformatics & Data Science Researcher* *Digital Egypt Builders Initiative (DEBI) Scholar*

---

```

```
