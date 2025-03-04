# Titanic- Machine Learning from Disaster
This repository contains a solution for the **"Titanic - Machine Learning from Disaster"** competition on Kaggle.

## 📌 Project Overview

The goal of this competition is to build a **predictive model** that determines which passengers were more likely to survive the **RMS Titanic** shipwreck. The dataset provides **demographic** and **socio-economic** information about passengers, allowing us to explore the factors that influenced survival rates.

## 🚀 Approach

The project follows a structured workflow:

1. **Data Preprocessing**  
   - Handling missing values (age, cabin, embarked port, etc.)  
   - Encoding categorical variables (gender, embarked port, etc.)  
   - Feature engineering (family size, name titles, cabin class)  

2. **Exploratory Data Analysis (EDA)**  
   - Identifying key survival trends  
   - Visualizing relationships between passenger attributes and survival rates  

3. **Model Selection & Training**  
   - Testing classification algorithms (Logistic Regression, Random Forest, XGBoost, etc.)  
   - Hyperparameter tuning for optimal performance  

4. **Prediction & Evaluation**  
   - Generating survival predictions for test data  
   - Evaluating model performance using **accuracy and confusion matrix**  

## 📊 Dataset

The dataset consists of:
- **train.csv**: Training data with survival labels  
- **test.csv**: Data for generating survival predictions  
- **gender_submission.csv**: Sample submission format
