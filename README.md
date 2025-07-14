# 🧠 Supervised Learning Project – Dual Domain Case Studies

This repository contains two applied Supervised Learning case studies:

- **Part A**: Medical data classification using biomechanical features
- **Part B**: Banking customer conversion prediction using behavioral and demographic data

Each part walks through the entire ML pipeline — from data understanding to model evaluation and performance improvement.

---

## ✅ PART A: Medical Classification – Predicting Patient Conditions

### 🏥 Objective
Predict the type of condition a patient is diagnosed with based on their biomechanical attributes. This is a multi-class classification task using three labeled datasets.

### 📁 Datasets
- `Normal.csv` – Patients without conditions
- `Type_H.csv` – Patients diagnosed with Condition Type H
- `Type_S.csv` – Patients diagnosed with Condition Type S

Each dataset includes:
- 6 biomechanical features: `P_incidence`, `P_tilt`, `L_angle`, `S_slope`, `P_radius`, `S_Degree`
- `Class`: Label indicating the condition

---

### 🔧 Workflow Summary

#### 1. Data Understanding
- Loaded all three datasets and checked their structure
- Ensured consistent column names and data types
- Observed class label inconsistencies (e.g., `Type_S`, `tp_s`)

#### 2. Data Cleaning & Preparation
- Normalized the `Class` column values to standard format: `type_s`, `type_h`, `normal`
- Merged the three datasets into one combined DataFrame
- Verified data integrity (null checks, summary statistics)

#### 3. Exploratory Data Analysis (EDA)
- Plotted a heatmap to visualize correlation between features
- Used pairplots to explore class separability
- Created a jointplot for `P_incidence` and `S_slope`
- Boxplots helped assess distribution and outliers

#### 4. Model Building
- Performed an 80:20 train-test split
- Trained a **K-Nearest Neighbors (KNN)** classifier
- Evaluated using accuracy, precision, recall, F1 score

#### 5. Performance Improvement
- Tuned `n_neighbors` and distance metrics
- Compared performance before and after tuning
- Documented improvements with metrics

---

### 📈 Key Takeaways
- EDA was critical for understanding class distributions
- Data label normalization was essential before model training
- KNN gave decent performance after hyperparameter tuning

---

## ✅ PART B: Banking Campaign – Predicting Credit Card Loan Conversion

### 🏦 Objective
Build a binary classifier to predict whether a customer will take a **loan on their credit card**, enabling targeted marketing campaigns.

### 📁 Datasets
- `Data1.csv` – Contains demographic and spending features
- `Data2.csv` – Contains banking behaviors like InternetBanking, CreditCard usage, etc.

Merged on the `ID` column.

---

### 🔧 Workflow Summary

#### 1. Data Understanding & Preparation
- Merged `Data1` and `Data2` using customer ID
- Changed appropriate binary/categorical features to type `object`
  - `CreditCard`, `InternetBanking`, `FixedDepositAccount`, `Security`, `Level`, `HiddenScore`

#### 2. Data Cleaning & Exploration
- Checked class balance of `LoanOnCard`
- Handled missing values and cleaned anomalous categorical values (`?`, `1.5`, etc.)
- Ensured clean, labeled, and usable data for modeling

#### 3. Model Building – Logistic Regression
- Dropped `ID` and `ZipCode`
- Performed 75:25 train-test split
- Trained a **Logistic Regression** model
- Evaluated performance with all key metrics

#### 4. Data Balancing
- Identified imbalance (e.g., 80:20 class distribution)
- Applied SMOTE/undersampling to balance to 50:50
- Re-trained the model and compared results before/after balancing

#### 5. Performance Improvement
- Trained additional classifiers: **KNN**, **SVM**
- Tuned hyperparameters using trial-and-error
- Compared all models and reported metrics

---

### 📈 Key Takeaways
- Data balancing improved recall for the minority class
- Logistic Regression performed well after cleaning
- KNN and SVM brought further improvements with tuning

---

## 🧰 Tools & Libraries Used
- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn`, `imbalanced-learn`
- `jupyter` for analysis and visualization

---

## 📚 Learning Outcomes
- Preprocessing pipelines for multi-source and multi-class datasets
- Visualization for exploratory analysis and feature relationships
- Hands-on experience with classification algorithms: Logistic Regression, KNN, SVM
- Understanding of class imbalance and its impact on model performance
- Practical model tuning using parameter trials

---

## 🚀 Getting Started

```bash

# Install required libraries
pip install -r requirements.txt
