# 💳 Customer Transaction Prediction

A Machine Learning project that predicts whether a bank customer is likely to make a transaction based on **200 anonymized numerical features**.

The project focuses on handling class imbalance, comparing multiple Machine Learning models, optimizing the best model, and selecting an appropriate decision threshold.

## 🚀 Features

* Data quality and integrity checks
* Missing value and duplicate analysis
* Class imbalance handling
* Stratified K-Fold cross-validation
* Model comparison using ROC-AUC
* Hyperparameter tuning with RandomizedSearchCV
* F1-optimal decision threshold selection
* Feature importance analysis
* Production-ready model serialization

## 🤖 Models Used

* Logistic Regression
* Random Forest Classifier
* LightGBM Classifier

The models are evaluated using **ROC-AUC**, which is suitable for this imbalanced classification problem.

The tuned **LightGBM model** achieved the best overall performance and was selected as the final model.

## ⚙️ Workflow

```text
Dataset
   ↓
Data Integrity Checks
   ↓
Train-Test Split
   ↓
Feature Scaling (Logistic Regression)
   ↓
Model Comparison
   ↓
Stratified Cross-Validation
   ↓
LightGBM Hyperparameter Tuning
   ↓
Threshold Optimization
   ↓
Model Evaluation
   ↓
Feature Importance
   ↓
Saved Model
```

## 📊 Evaluation Metrics

The project uses:

* ROC-AUC
* Precision
* Recall
* F1-Score
* Confusion Matrix

An F1-optimal threshold is selected instead of using the default probability threshold of `0.5`.

## 💾 Saved Models

The final artifacts include:

```text
best_customer_transaction_model.pkl
decision_threshold.pkl
```

These files can be used for future predictions and deployment.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* LightGBM
* Matplotlib
* Seaborn
* Joblib

## 📂 Project Structure

```text
Customer-Transaction-Prediction/
│
├── main.ipynb
├── train.csv
├── best_customer_transaction_model.pkl
├── decision_threshold.pkl
├── README.md
└── requirements.txt

