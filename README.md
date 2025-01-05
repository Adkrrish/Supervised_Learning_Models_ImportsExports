# Supervised Machine Learning: Classification & Regression Analysis

## Project Overview
This project applies supervised machine learning algorithms to analyze and classify trade-related data. The primary objective is to evaluate the performance of classification models, identify important features, and derive actionable insights. Models such as Logistic Regression, Decision Trees, and others are implemented and compared based on their performance metrics.

---

## Problem Statement
The goal of this project is to classify trade transactions into distinct classes or categories based on various trade-related attributes. This classification helps uncover patterns and trends that can drive strategic business decisions. The key objectives include:

- Classifying trade transactions using supervised learning models.
- Identifying important features and determining thresholds for classification.
- Comparing models to determine the most suitable one for the dataset.

---

## Key Features
1. **Classification Models**:
   - Logistic Regression (LR)
   - Support Vector Machines (SVM)
   - Stochastic Gradient Descent (SGD)
   - Decision Trees (DT)
   - K-Nearest Neighbors (KNN)
   - Random Forest (RF)
   - Naive Bayes (NB)
   - XGBoost (Extreme Gradient Boosting)
2. **Performance Evaluation**:
   - Accuracy, Precision, Recall, F1-Score, AUC
   - Cross-Validation (K-Fold) for robust model evaluation
3. **Run Statistics**:
   - Training time and memory usage for all models
4. **Feature Importance**:
   - Identification of significant features like `Country`, `Product`, and `Value`
   - Thresholds and business implications of key features
5. **Data Preprocessing**:
   - Min-Max scaling for numeric data
   - Ordinal encoding for categorical data

---

## Dataset
**Source**: [Kaggle Import-Export Dataset](https://www.kaggle.com/datasets/chakilamvishwas/imports-exports-15000)  
**Sample Size**: 5,001 rows selected from 15,000 records  
**Key Variables**:
- **Categorical**: `Country`, `Product`, `Import_Export`, `Category`, `Port`, `Shipping_Method`, `Supplier`, `Customer`, `Payment_Terms`
- **Numerical**: `Quantity`, `Value`, `Weight`
- **Index Variables**: `Transaction_ID`, `Invoice_Number`

---

## Technologies Used
- **Programming Language**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Environment**: Google Colab
- **Version Control**: GitHub (for version tracking and collaboration)

---

## Methodology
1. **Data Preprocessing**:
   - Data Cleaning: No missing data; no treatment required.
   - Encoding: Ordinal Encoder for categorical variables.
   - Scaling: Min-Max Scaling for numerical data.
2. **Classification Models**:
   - Training and testing split (70:30) applied to evaluate models.
   - Models compared based on their test set performance and cross-validation results.
3. **Performance Metrics**:
   - Confusion Matrix
   - Precision, Recall, F1-Score, AUC
   - Training time and memory usage for runtime analysis.
4. **Feature Analysis**:
   - Identification of significant features using Random Forest and Logistic Regression.
   - Thresholds derived for key features like `Value`, `Weight`, and `Shipping_Method`.

---

## Insights and Applications
1. **Model Insights**:
   - Logistic Regression performed well in runtime and interpretability but struggled with class imbalance.
   - Random Forest provided robust feature importance insights but was resource-intensive.
   - SVM and SGD focused heavily on dominant classes, highlighting the need for balanced datasets.
2. **Business Applications**:
   - **Targeted Marketing**: Use key insights from clusters to create marketing strategies.
   - **Inventory Management**: Focus on trade-specific patterns like bulk vs. lightweight goods.
   - **Supplier-Customer Relations**: Optimize Payment Terms and Shipping Methods to enhance trade efficiency.

---

## Contributors
- Krishnendu Adhikary
