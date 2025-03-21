# Wrapper Feature Selection (Part-2)

## Overview
This project applies **Wrapper Feature Selection (Part-2)** using **Recursive Feature Elimination with Cross-Validation (RFECV)** to identify the most important features for predicting wine quality. The dataset is obtained from the **UCI Machine Learning Repository**.

## Dataset
The dataset used is **Wine Quality Dataset** from UCI ML Repository, containing various physicochemical properties of wine samples and their quality ratings.

## Installation
To run this project, ensure you have Python installed along with the required dependencies.

## code

### Import necessary libraries 
```
import pandas as pd
from sklearn.feature_selection import RFECV 
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
```
### Load dataset 
```
df = pd.read_csv('/content/drive/MyDrive/Datasets/winequality-red.csv', delimiter=';') 
```
### Separate features and target variable 
```
X = df.drop(columns=['quality']) 
y = df['quality'] 
```
### Standardize the data to improve model performance 
```
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X) 
```
### Split data into training and testing sets 
```
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42) 
```
### Initialize logistic regression model 
```
model = LogisticRegression(max_iter=2000)
```
### Apply RFECV (Recursive Feature Elimination with Cross-Validation) 
```
rfecv = RFECV(estimator=model, step=1, cv=5) 
rfecv.fit(X_train, y_train) 
```
### Get selected feature names 
```
selected_features = X.columns[rfecv.support_] 
print("Optimal number of features:", rfecv.n_features_) 
print("Selected Features using RFECV:", selected_features.tolist())
```
