# Feature Selection Using Correlation on Wine Quality Dataset

## **Overview**
This repository demonstrates **filter-based feature selection** using the **UCI Wine Quality Dataset**.  
Filter methods use **statistical techniques** to select relevant features before training a machine learning model.

## **Dataset**
The dataset used is the **Wine Quality Dataset** from the **UCI ML Repository**.  
It contains **chemical properties** of wine and their corresponding **quality ratings**.

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)
- **Features**:
  - Fixed acidity
  - Volatile acidity
  - Citric acid
  - Residual sugar
  - Chlorides
  - Free sulfur dioxide
  - Total sulfur dioxide
  - Density
  - pH
  - Sulphates
  - Alcohol
- **Target Variable**:  
  - `quality` (Wine quality score between **0 and 10**)

## **Techniques Used**
1. **Correlation-Based Feature Selection**  
   - Computes a **correlation matrix** between all features.
   - Selects features that have a **correlation above 0.3** with the target variable (`quality`).
   - Removes features that have a low impact on wine quality.

## code
```
### Import necessary libraries
import pandas as pd
<br>
import numpy as np
<br>
import seaborn as sns
<br>
import matplotlib.pyplot as plt

### Load dataset (example: UCI Wine Quality dataset)
df = pd.read_csv('/content/drive/MyDrive/Datasets/winequality-red.csv', delimiter=';')

### Compute correlation matrix
corr_matrix = df.corr()

### Plot heatmap of correlation matrix
plt.figure(figsize=(10, 6))
<br>
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
<br>
plt.title("Feature Correlation Heatmap")
<br>
plt.show()

### Select features with high correlation (above a threshold)
correlation_threshold = 0.3
<br>
target_variable = 'quality'
<br>
correlated_features = corr_matrix[target_variable][abs(corr_matrix[target_variable]) > correlation_threshold].index.tolist()
<br>
correlated_features.remove(target_variable)

### Display selected features
print("Selected features based on correlation:", correlated_features)

### Create new dataframe with selected features
filtered_df = df[correlated_features + [target_variable]]
<br>
print("\nShape of filtered dataset:", filtered_df.shape)


## **Installation**
To run the code, install the required libraries:
```bash
pip install pandas numpy seaborn matplotlib


