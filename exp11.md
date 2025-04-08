#  Wrapper-Based Feature Selection on the Wine Dataset

This project demonstrates how to apply **wrapper-based feature selection methods** on the **Wine dataset** using multiple machine learning models.

Wrapper methods evaluate subsets of features based on model performance, offering a tailored approach to feature selection.

## overview

This project explores how to identify the most relevant features from the Wine dataset using wrapper methods of feature selection. Wrapper methods are model-based approaches that evaluate subsets of features by training a model and measuring its performance — offering a tailored selection compared to filter or embedded methods.
---

## Dataset

- **Wine Dataset** from `sklearn.datasets.load_wine`.
- Multi-class classification task with **13 numerical features**.
- Goal: Classify wine types based on chemical analysis.

---

## Feature Selection Methods Used

### 1.  Recursive Feature Elimination (RFE)

- Removes least important features recursively based on model weights.
- Model: `LogisticRegression`

### 2.  Sequential Feature Selector (SFS)

- Greedy search (forward or backward) using cross-validation.
- Library: `mlxtend`
- Model: `RandomForestClassifier`

### 3.  Exhaustive Feature Selector (Optional)

- Tries **all** possible combinations for a specified number of features.
- Useful when feature space is small.
- Model: `LogisticRegression`

---

### code
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load data
data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Standardize for Lasso
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# LASSO 
lasso = LogisticRegression(penalty='l1', solver='saga', multi_class='multinomial', max_iter=5000, C=1.0)
lasso.fit(X_scaled, y)

# Get non-zero coefficients (feature selection)
lasso_coefs = np.mean(np.abs(lasso.coef_), axis=0)
lasso_mask = lasso_coefs != 0
lasso_features = X.columns[lasso_mask]
lasso_importance = lasso_coefs[lasso_mask]

# RANDOM FOREST
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)
rf_importances = rf.feature_importances_
rf_features = X.columns
rf_importance_sorted_idx = np.argsort(rf_importances)[::-1]
top_rf_features = rf_features[rf_importance_sorted_idx]
top_rf_importance = rf_importances[rf_importance_sorted_idx]

# PLOT BOTH 
plt.figure(figsize=(15, 6))

# Lasso Plot
plt.subplot(1, 2, 1)
bars1 = plt.bar(lasso_features, lasso_importance, color='orange')
plt.title("Lasso - Feature Importance")
plt.ylabel("Coefficient Magnitude")
plt.xticks(rotation=90)
for bar in bars1:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.01, f"{height:.3f}", ha='center', va='bottom', fontsize=8)
plt.ylim(0, max(lasso_importance)*1.2)

# Random Forest Plot
plt.subplot(1, 2, 2)
bars2 = plt.bar(top_rf_features, top_rf_importance, color='green')
plt.title("Random Forest - Feature Importance")
plt.ylabel("Importance Score")
plt.xticks(rotation=90)
for bar in bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.005, f"{height:.3f}", ha='center', va='bottom', fontsize=8)
plt.ylim(0, max(top_rf_importance)*1.2)

plt.tight_layout()
plt.show()
```
### output

![exp11](https://github.com/user-attachments/assets/951284bb-5267-4643-a26e-6332978bd29e)
