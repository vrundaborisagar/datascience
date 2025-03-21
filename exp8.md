# Feature Selection Using Chi-Square Test on Wine Quality Dataset

## **Overview**
This repository demonstrates **filter-based feature selection** using the **Chi-Square test** on the **UCI Wine Quality Dataset**.

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
1. **Chi-Square Test for Feature Selection**  
   - Measures the dependence between **features** and **target variable (`quality`)**.  
   - Selects the **top K features** with the highest dependency on the target.

# code
```
### Import necessary libraries
import pandas as pd
</br>
from sklearn.feature_selection import SelectKBest, chi2

### Load dataset
df = pd.read_csv('/content/drive/MyDrive/Datasets/winequality-red.csv', delimiter=';')

### Separate features and target variable
X = df.drop(columns=['quality'])
</br>
y = df['quality']

### Apply SelectKBest with Chi-Square Test
k = 5  # Select top 5 features
</br>
chi_selector = SelectKBest(score_func=chi2, k=k)
</br>
X_selected = chi_selector.fit_transform(X, y)

### Get selected feature names
selected_feature_indices = chi_selector.get_support(indices=True)
</br>
selected_features = X.columns[selected_feature_indices]
</br>
print("Top", k, "selected features using Chi-Square Test:", selected_features.tolist())

## **Installation**
To run the code, install the required libraries:
```bash
pip install pandas scikit-learn
