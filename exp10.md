# Wrapper Feature Selection (Part-2)

## Overview
This project applies **Wrapper Feature Selection (Part-2)** using **Recursive Feature Elimination with Cross-Validation (RFECV)** to identify the most important features for predicting wine quality. The dataset is obtained from the **UCI Machine Learning Repository**.

## Dataset
The dataset used is **Wine Quality Dataset** from UCI ML Repository, containing various physicochemical properties of wine samples and their quality ratings.

## Installation
To run this project, ensure you have Python installed along with the required dependencies.

## code
```
### Import necessary libraries <br>
import pandas as pd <br>
from sklearn.feature_selection import RFECV <br>
from sklearn.linear_model import LogisticRegression <br>
from sklearn.model_selection import train_test_split <br>
from sklearn.preprocessing import StandardScaler <br>

### Load dataset <br>
df = pd.read_csv('/content/drive/MyDrive/Datasets/winequality-red.csv', delimiter=';') <br>

### Separate features and target variable <br>
X = df.drop(columns=['quality']) <br>
y = df['quality'] <br>

### Standardize the data to improve model performance <br>
scaler = StandardScaler() <br>
X_scaled = scaler.fit_transform(X) <br>

### Split data into training and testing sets <br>
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42) <br>

### Initialize logistic regression model <br>
model = LogisticRegression(max_iter=2000) <br>

### Apply RFECV (Recursive Feature Elimination with Cross-Validation) <br>
rfecv = RFECV(estimator=model, step=1, cv=5) <br>
rfecv.fit(X_train, y_train) <br>

### Get selected feature names <br>
selected_features = X.columns[rfecv.support_] <br>
print("Optimal number of features:", rfecv.n_features_) <br>
print("Selected Features using RFECV:", selected_features.tolist()) <br>
