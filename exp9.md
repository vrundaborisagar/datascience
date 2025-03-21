# Wrapper Feature Selection (RFE)

## Overview  
Feature selection is a crucial step in machine learning that helps improve model performance by eliminating irrelevant or redundant features.  
This project demonstrates the **Wrapper Feature Selection** technique using **Recursive Feature Elimination (RFE)** with **Logistic Regression**.  
The goal is to select the top **5 most significant features** from the **Wine Quality Dataset** obtained from the **UCI ML Repository**.  

## Description  
Recursive Feature Elimination (RFE) is a wrapper method that selects the most relevant features by recursively training a model and eliminating the least important features.  
In this project, **Logistic Regression** is used as the base model for RFE to determine the most influential features in predicting wine quality.  

## How to Use  
1. Download the dataset and save it in the correct location.  
2. Copy and run the Python script in your local environment or a Jupyter Notebook.  
3. Ensure you have installed the required libraries (`pandas`, `sklearn`).  
4. The script will output the **top 5 selected features**.  

# Code  

## Import necessary libraries </br>
```
import pandas as pd </br>
from sklearn.feature_selection import RFE </br>
from sklearn.linear_model import LogisticRegression </br>
from sklearn.model_selection import train_test_split </br>
```
## Load dataset 
```
df = pd.read_csv('/content/drive/MyDrive/Datasets/winequality-red.csv', delimiter=';') 
```

## Separate features and target variable </br>
```
X = df.drop(columns=['quality']) 
y = df['quality'] 
```

## Split data into training and testing sets 
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
```
## Initialize logistic regression model 
```
model = LogisticRegression(max_iter=5000) 
```
# Increased max_iter to ensure convergence
```
## Apply RFE (Recursive Feature Elimination) to select top 5 features
rfe = RFE(estimator=model, n_features_to_select=5) 
X_selected = rfe.fit_transform(X_train, y_train)
```
## Get selected feature names
```
selected_features = X.columns[rfe.support_] 
print("Selected Features using RFE:", selected_features.tolist()) 
```
