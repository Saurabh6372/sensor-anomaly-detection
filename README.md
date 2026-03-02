# Sensor-Based Anomaly Detection  

## Overview

This project focuses on detecting anomalies from multi-sensor time-series data using machine learning techniques. The objective was to maximise F1-score through advanced feature engineering, imbalance handling, and threshold optimisation.

---

## Problem

Given sensor readings captured over time, predict whether a record represents anomalous behaviour.

Challenges:
- Severe class imbalance
- Temporal dependency
- Outliers in sensor values
- Threshold optimisation for F1

---

## Approach

### Feature Engineering
- Extracted hour, day-of-week, and month from datetime
- Created interaction feature: `X1_X2_ratio`
- Applied outlier clipping
- Robust scaling

### Imbalance Handling
- SMOTE oversampling (training folds only)
- scale_pos_weight tuning

### Modeling
- XGBoost
- Random Forest (experimental)
- TimeSeriesSplit cross-validation
- Precision-Recall based threshold tuning

### Evaluation
Optimised for:
- F1 Score
- Class-wise performance
- Robustness via cross-validation

---

## Tech Stack
- Python
- Scikit-learn
- XGBoost
- SMOTE
- SHAP
- Pandas / NumPy

---

## How to Run

1. Install dependencies: pip install -r requirements.txt

2. Download dataset from Kaggle

3. Place data inside:  data/
train.parquet
test.parquet


4. Run the code 
---

## Key Learnings
- Threshold tuning significantly improves F1-score
- Time-aware validation prevents leakage
- Proper imbalance handling is critical in anomaly detection

---

Author: Saurabh Maurya  
MSc Computer Science – University of Greenwich
