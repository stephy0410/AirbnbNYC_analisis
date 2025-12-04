# 🗽 Airbnb NYC 2019: Price Analysis & Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange)

## 📖 Project Overview
This project focuses on analyzing the **New York City Airbnb 2019** dataset to understand the factors influencing accommodation prices. By applying Data Mining techniques, we cleaned, transformed, and modeled the data to predict rental prices based on physical and geographical characteristics.

The project includes a **Streamlit Web Application** that visualizes the data processing stages (Dirty vs. Clean data) and the results of the predictive models.

## 🎯 Objectives
* **Business Problem:** Identifying the key factors that determine property prices to help hosts set competitive rates and help travelers understand cost drivers.
* **Data Mining Goal:** To model rental prices using characteristics like neighborhood, room type, and availability, identifying patterns to optimize price setting.

## 👥 Team Members
* Stephanie Borrego Arroyo
* Hannah Chenoa Puente Rosales
* Yana Elina Medina García
* Luis Francisco Zárate Díaz

## 📊 Data Source
The dataset used is **New York City Airbnb Open Data (2019)** from [Inside Airbnb](http://insideairbnb.com/), available on Kaggle.
* **Records:** ~48,895
* **Features:** 16 (including price, location, room type, reviews, etc.)

## 🛠️ Methodology & Workflow

### 1. Data Understanding & Cleaning
We addressed data quality issues to ensure accurate modeling:
* **Handling Nulls:** Filled missing values in `reviews_per_month` with `0` (implying no reviews) and removed rows with missing names/host names.
* **Removing Redundancy:** Dropped the `last_review` column as it was redundant with `reviews_per_month`.
* **Outlier Removal:** Filtered out unrealistic prices (e.g., $0 or >$500) to reduce skewness.
* **Memory Optimization:** Converted data types (e.g., `float64` to `float32`, `object` to `category`) for better performance.

### 2. Data Transformation
* **Discretization:** Grouped continuous variables like `price` and `minimum_nights` into categories (e.g., "Economic", "Moderate", "Short Stay").
* **Mapping:** Converted categorical variables (`neighbourhood_group`, `room_type`) into numerical values for correlation analysis and modeling.

### 3. Modeling
We implemented and compared two regression models to predict prices:
1.  **Multiple Linear Regression:** Used as a baseline to quantify linear relationships.
2.  **k-Nearest Neighbors (k-NN):** Captures non-linear patterns based on local similarity.

## 📈 Results
After evaluating both models, **k-NN** proved to be superior for this specific dataset.

| Metric | Linear Regression | k-NN (k=21) |
| :--- | :--- | :--- |
| **R² (Determination Coeff)** | 0.2566 | **0.3489** |
| **RMSE (Root Mean Sq Error)** | 101.79 | **95.26** |
| **MAE (Mean Absolute Error)** | 60.69 | **54.16** |

*Conclusion:* The k-NN model ($k=21$) provided a better fit ($R^2 \approx 35\%$) and lower error rates compared to Linear Regression, suggesting that local similarity (neighborhood characteristics) is a strong predictor of price.

## 🚀 How to Run the Project

### Prerequisites
Ensure you have Python installed. Install the required libraries:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn statsmodels streamlit
