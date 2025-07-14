# 🧠 Supervised Learning Project – Patient Condition Classification

## 📘 Overview
This project explores the application of **Supervised Learning** to a classification problem in the medical domain. The data represents biomechanical measurements of patients, and the objective is to predict their condition category using machine learning models. This project walks through a complete ML workflow: from loading and cleaning the data to visualizing and building models.

---

## 📁 Dataset
There are 3 CSV files representing different patient condition categories:
- `Normal.csv`: Biomechanical features of patients without any diagnosed condition.
- `Type_H.csv`: Patients with condition type H.
- `Type_S.csv`: Patients with condition type S.

Each file contains 7 columns:
- `P_incidence`, `P_tilt`, `L_angle`, `S_slope`, `P_radius`, `S_Degree` — numerical biomechanical features
- `Class` — target variable (condition label)

---

## ✅ PART A: Medical Classification

### 🔍 1. Data Understanding
- Loaded all 3 datasets and checked their shape (rows and columns).
- Confirmed that all datasets have consistent columns.
- Inspected the data types of each feature.
- Identified variations in the `Class` column such as `'Normal'`, `'type_S'`, `'Type_S'`, `'tp_s'` etc.

### 🧹 2. Data Preparation & Cleaning
- Normalized class names to a standard format (`type_s`, `type_h`, `normal`).
- Concatenated the three dataframes into one unified dataset.
- Displayed sample records, checked for null values, and summarized statistical characteristics.

### 📊 3. Exploratory Data Analysis (EDA)
- Used **heatmaps** to analyze feature correlation.
- **Pairplots** were used to visualize class separability.
- Created a **jointplot** for `P_incidence` vs `S_slope` and inferred clustering by class.
- **Boxplots** were generated for each feature to inspect outliers and distribution spread.

### 🤖 4. Model Building
- Separated features (`X`) and target (`y`).
- Split the data into training and testing sets (80:20 split).
- Trained a **K-Nearest Neighbors (KNN)** classifier.
- Evaluated model using accuracy, precision, recall, and F1 score on both train and test data.

### 🚀 5. Performance Improvement
- Tuned KNN hyperparameters (e.g., `n_neighbors`) to improve performance.
- Reported metric improvements like accuracy uplift.
- Identified which parameters had the most impact.

---

## 🧰 Tools Used
- **Python** (pandas, numpy, matplotlib, seaborn, scikit-learn)
- **Jupyter Notebook**

---

## 📚 Key Learnings
- Data cleaning and normalization is essential when working with datasets from multiple sources.
- Visualization can offer powerful insights into feature relationships and class separability.
- Simple models like KNN can perform well with good preprocessing and parameter tuning.
- Evaluation on both training and test data is crucial to avoid overfitting.

---

## 🚀 Getting Started
```bash

# Install dependencies
pip install -r requirements.txt

```

---

## 📌 Feel Free to Download and Use!
- 
