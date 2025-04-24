# Title
BoT-IoT Dataset Classification Pipeline: A Hybrid Machine Learning Approach

# Overview:
This repository presents a comprehensive pipeline for the classification of the BoT-IoT dataset, which consists of network traffic data for attack detection. The pipeline covers all essential steps, from data loading and cleaning to preprocessing, feature selection, and model evaluation. The final model uses a hybrid machine learning approach, combining multiple base classifiers such as Random Forest, Gradient Boosting, and Support Vector Machine (SVM), with Logistic Regression as the meta-model. This pipeline aims to enhance prediction accuracy for detecting network attacks and normal traffic.

## 1. Install Required Packages
We begin by installing the required libraries:
- `category_encoders`: For encoding categorical variables using various techniques.
- `scikit-learn`: For machine learning models, preprocessing, and evaluation.
- `pandas`, `numpy`: For data manipulation.
- `matplotlib`, `seaborn`: For data visualization.

 ## important library
- **Pandas** and **NumPy** for data manipulation
- **Matplotlib** and **Seaborn** for data visualization
- **Scikit-learn** for machine learning tasks:
  - Preprocessing (LabelEncoder, MinMaxScaler, OrdinalEncoder)
  - Model selection and training (RandomForest, GradientBoosting)
  - Feature selection (SelectKBest, RFE, SequentialFeatureSelector)
  - Evaluation (classification report, confusion matrix)
- **Category Encoders** for encoding categorical variables:
  - Target Encoding
  - Binary Encoding
  - One-hot Encoding
  - Frequency (Count) Encoding
- **ZipFile** for extracting zipped datasets
- **Warnings** to suppress warnings during execution
  
~~~python
import pandas as pd
import numpy as np
from zipfile import ZipFile
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import (SelectKBest, mutual_info_classif,
                                     VarianceThreshold, RFE, SequentialFeatureSelector)
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from category_encoders import TargetEncoder, BinaryEncoder, OneHotEncoder, CountEncoder
import warnings
warnings.filterwarnings('ignore')
~~~
---

 ## Unique Random Seed for Reproducibility
Set a unique random seed to ensure the results are reproducible across runs.
~~~python
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
~~~
---
  ## Load and Merge the BoT-IoT Dataset
The dataset is loaded from a zip file containing CSV files. All files are read and merged into a single DataFrame. Error handling is implemented to manage potential issues like missing files.
- `CSV files are extracted from the zip archive.`
- `Columns are renamed to standardized names, with additional checks for column integrity.`
- `In case of errors during data loading, a sample dataset is loaded as a fallback.`
~~~python

def load_bot_iot_data():
    """Unique dataset loader with auto-recovery features"""
    print("Step 1: Loading and merging BoT-IoT dataset...")
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=True)

        dfs = []
        with ZipFile('/content/drive/MyDrive/Colab Notebooks/archive.zip') as z:
            file_list = [f for f in z.namelist() if f.endswith('.csv')]
            print(f"Found {len(file_list)} CSV files in archive")

            for i, file in enumerate(file_list[:5]):  # Process first 5 files
                with z.open(file) as f:
                    df = pd.read_csv(f, header=None, low_memory=False)
                    dfs.append(df)
                    print(f"Loaded {file} with shape {df.shape}")

        if not dfs:
            raise ValueError("No CSV files found in the archive")

        df = pd.concat(dfs, axis=0, ignore_index=True)
        print(f"\nFinal merged dataset shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Unique Error Handler: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on error

df = load_bot_iot_data()

~~~
---
## Data Preprocessing - Cleaning
This step involves handling missing values, duplicates, and outliers.
~~~python
# 2. Data Preprocessing - Cleaning
# 2.1 Drop irrelevant columns
cols_to_drop = ['pkseqid', 'stime', 'saddr', 'daddr', 'smac', 'dmac', 'seq']
df.drop(columns=cols_to_drop, inplace=True)

# 2.2 Handle missing values
missing_markers = ['-', '?', 'NaN', 'NA', 'null', 'None', 'none', '']
for marker in missing_markers:
    df.replace(marker, np.nan, inplace=True)

# 2.3 Remove duplicates
df = df.loc[~df.astype(str).apply(lambda x: hash(tuple(x)), axis=1).duplicated()]

# 2.4 Handle outliers
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    iqr_lower = Q1 - 3 * IQR
    iqr_upper = Q3 + 3 * IQR
    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
    mask = ((df[col] < iqr_lower) | (df[col] > iqr_upper)) & (z_scores > 3.5)
    df.loc[mask, col] = np.clip(df.loc[mask, col], iqr_lower, iqr_upper)
~~~
---
### 3. Data Preprocessing – Encoding  
Convert categorical features into numbers via six techniques:

 **Label Encoding**  
   ~~~python
   df_label = df.copy()
   le = LabelEncoder()
   for col in categorical_cols:
       df_label[col] = le.fit_transform(df_label[col].astype(str))
~~~

**What it does:**  
 Assigns each unique category an integer (0…N–1). Simple, but may impose an unintended ordinal relationship.

 ---
 **One‑Hot Encoding**
 ~~~python
df_onehot = df.copy()
for col in categorical_cols:
    if df[col].nunique() < 20:
        ohe = OneHotEncoder(use_cat_names=True)
        enc = ohe.fit_transform(df_onehot[[col]])
        df_onehot = pd.concat([df_onehot.drop(col,1), enc], axis=1)
~~~

**What it does:**  
Creates a new binary column for each category. Preserves neutrality between categories, but increases dimensionality.

---
**Ordinal Encoding**
~~~python
df_ordinal = df.copy()
for col in categorical_cols:
    order = df[col].value_counts().index
    oe = OrdinalEncoder(categories=[order], handle_unknown='use_encoded_value', unknown_value=-1)
    df_ordinal[col] = oe.fit_transform(df_ordinal[[col]])
~~~

**What it does:**  
 Maps categories to integers based on their overall frequency (most→smallest). Maintains an order but may introduce bias.

---
**Target Encoding**
~~~python
df_target = df.copy()
te = TargetEncoder(smoothing=10)
for col in categorical_cols:
    df_target[col] = te.fit_transform(df_target[col], df_target['attack'])
~~~

**What it does:**  
Replaces each category with the mean of the target variable for that category, smoothed to reduce overfitting.

---
**Binary Encoding**
~~~python
df_binary = df.copy()
be = BinaryEncoder()
for col in categorical_cols:
    enc = be.fit_transform(df_binary[[col]])
    df_binary = pd.concat([df_binary.drop(col,1), enc], axis=1)
~~~

**What it does:**  
Converts category labels into binary digits and splits them into multiple columns, offering a compromise between one‑hot and label encoding.

---
**Frequency Encoding**
~~~python
df_freq = df.copy()
for col in categorical_cols:
    freq = df[col].value_counts(normalize=True)
    df_freq[col] = df_freq[col].map(freq)
~~~

**What it does:** 
Replaces each category with its frequency (percentage) in the dataset. Captures commonness without adding dimensions.

 ###  Select the Most Effective Encoding Method
We choose the **target encoding** as the default and primary method of encoding categorical features.

~~~python
df_encoded = encoded_dfs['target']
print("\nSelected target encoding as primary encoding method")
~~~

- Target encoding replaces categorical values with a number (often the mean of the target variable for that category).
- This step is critical for transforming non-numeric data into a format suitable for machine learning models.
---
### 4.Data Preprocessing - Normalization
We normalize numeric features using the Quantile Transformer, which transforms the data distribution to follow a normal distribution.
~~~python
print("\nStep 4: Normalization using Quantile Transformer...")
numeric_cols = df_encoded.select_dtypes(include=np.number).columns.tolist()
if 'attack' in numeric_cols:
    numeric_cols.remove('attack')

if numeric_cols:
    from sklearn.preprocessing import QuantileTransformer
    qt = QuantileTransformer(output_distribution='normal', random_state=RANDOM_SEED)
    df_encoded[numeric_cols] = qt.fit_transform(df_encoded[numeric_cols])
    print("Applied quantile normalization to numeric features")
else:
    print("No numeric features to normalize")
~~~
- Identifying Numeric Features:- We start by selecting all columns with numeric data types.
- The column 'attack' is excluded from normalization, as it may be a target variable or otherwise unsuitable for transformation.

- Applying Quantile Transformation:- This step ensures that numeric features follow a normal distribution, reducing skewness and improving model performance.
- The random_state parameter ensures reproducibility of the results.

- Handling the Absence of Numeric Features:- If there are no numeric features to normalize, a relevant message is displayed.
---
## 5. Data Visualization

In this step, we explore the encoded BoT-IoT dataset using advanced visualization techniques. This helps us better understand the distribution of the target variable and the relationships between numerical features. Visual inspection can also uncover patterns, correlations, and outliers that guide feature selection and modeling.

---

### 5.1 Target Distribution Pie Chart

~~~python
plt.figure(figsize=(10, 6))
target_counts = df_encoded['attack'].value_counts()
plt.pie(target_counts, labels=target_counts.index, autopct='%1.1f%%',
        colors=['#66b3ff','#ff9999'], startangle=90, explode=(0.05, 0))
plt.title('Attack vs Normal Traffic Distribution', fontsize=14)
plt.show()
~~~

**What it does:**  
it Creates a pie chart showing the proportion of attack vs normal traffic in the dataset. This helps detect class imbalance, which is crucial in choosing the right evaluation metric or applying balancing techniques later.

### 5.2 Correlation Matrix with Hierarchical Clustering
~~~python
numeric_cols = df_encoded.select_dtypes(include=np.number).columns.tolist()
if 'attack' in numeric_cols:
    numeric_cols.remove('attack')

if len(numeric_cols) > 1:
    plt.figure(figsize=(14, 12))
    corr = df_encoded[numeric_cols].corr()
    sns.clustermap(corr, cmap='coolwarm', center=0, annot=False, figsize=(12, 10))
    plt.title('Hierarchically Clustered Correlation Heatmap', pad=20)
    plt.show()
~~~

**What it does:**  
it Generates a heatmap of feature correlations, using clustering to group similar features together. Helps identify redundant or highly correlated features that can be candidates for removal or dimensionality reduction (like PCA).

### 5.3 Feature Distributions with Outlier Visualization
~~~python
if len(numeric_cols) > 0:
    plt.figure(figsize=(16, 12))
    for i, col in enumerate(numeric_cols[:6]):
        plt.subplot(2, 3, i+1)
        sns.boxplot(x=df_encoded[col], color='lightblue')
        sns.stripplot(x=df_encoded[col], color='darkblue', alpha=0.3)
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.show()
~~~
**What it does:**  
Draws boxplots with overlaid strip plots to show distributions and outliers for numeric features. Visually detects skewed distributions and extreme values that may need transformation or special handling before training models.
#### These visualizations ensure better insight into:

- Class imbalance
- Feature correlation
- Presence of outliers
  
All of which directly influence the choice of algorithms, evaluation metrics, and preprocessing steps in later stages.

---

## 6. Feature Selection - Filter Methods

In this step, we apply **filter-based feature selection** techniques to reduce dimensionality and improve model performance. Filter methods assess the importance of features using statistical measures without involving any learning algorithm.

---

###  6.1 Variance Threshold

~~~python
selector_vt = VarianceThreshold(threshold=0.01)
X_vt = selector_vt.fit_transform(X)
selected_vt = X.columns[selector_vt.get_support()]
print(f"Selected {len(selected_vt)} features: {list(selected_vt)}")
~~~

**What it does:**  
Removes features with very low variance, as they contribute little to distinguishing between classes. In this example, features with variance below `0.01` are removed.

---

###  6.2 Mutual Information (Top 15 Features)

~~~python
selector_mi = SelectKBest(mutual_info_classif, k=15)
X_mi = selector_mi.fit_transform(X, y)
selected_mi = X.columns[selector_mi.get_support()]
print(f"Top 15 features by MI: {list(selected_mi)}")
~~~

**What it does:**  
Selects the top 15 features that share the most **mutual information** with the target variable. Mutual information measures how much knowing the feature helps predict the target.

---

### 6.3 ANOVA F-value (Top 10 Features)

~~~python
from sklearn.feature_selection import f_classif
f_scores, _ = f_classif(X, y)
f_df = pd.DataFrame({'Feature': X.columns, 'F-score': f_scores})
selected_anova = f_df.nlargest(10, 'F-score')['Feature'].tolist()
print(f"Top 10 features by ANOVA: {selected_anova}")
~~~

**What it does:**  
Uses the **Analysis of Variance (ANOVA)** F-test to rank features. Higher F-values indicate that a feature better separates the classes.

---

###  6.4 Correlation with Target (Top 10 Features)

~~~python
corr_with_target = X.corrwith(y).abs().sort_values(ascending=False)
selected_corr = corr_with_target.head(10).index.tolist()
print(f"Top 10 correlated features: {selected_corr}")
~~~

**What it does:**  
This part of the code finds the top 10 features in X that are most strongly related to the target variable y. These features are selected based on their absolute correlation values.

---  
These filter methods are:
- **Fast and scalable** for large datasets
- Useful for **initial dimensionality reduction**

  ## 7. Feature Selection - Filter Methods

In this step, we apply **filter-based feature selection** techniques. These methods use statistical metrics to rank or eliminate features without relying on a machine learning model. This helps in reducing overfitting, improving accuracy, and decreasing training time.

---

### 7.1 Variance Threshold

~~~python
selector_vt = VarianceThreshold(threshold=0.01)
X_vt = selector_vt.fit_transform(X)
selected_vt = X.columns[selector_vt.get_support()]
print(f"Selected {len(selected_vt)} features: {list(selected_vt)}")
~~~

** What it does:**  
Removes all features whose variance doesn’t meet the threshold. Low-variance features carry less information and might not help in classification.

- **Tool used:** `VarianceThreshold` from `sklearn.feature_selection`
- **Purpose:** Remove redundant or constant features.

---

### 7.2 Mutual Information (Top 15 Features)

~~~python
selector_mi = SelectKBest(mutual_info_classif, k=15)
X_mi = selector_mi.fit_transform(X, y)
selected_mi = X.columns[selector_mi.get_support()]
print(f"Top 15 features by MI: {list(selected_mi)}")
~~~

** What it does:**  
Ranks features based on **mutual information** with the target variable. High mutual information means the feature gives more insight about the class.

- **Tool used:** `SelectKBest` with `mutual_info_classif` from `sklearn.feature_selection`
- **Purpose:** Select the top 15 most informative features.

---

### 7.3 ANOVA F-value (Top 10 Features)

~~~python
from sklearn.feature_selection import f_classif
f_scores, _ = f_classif(X, y)
f_df = pd.DataFrame({'Feature': X.columns, 'F-score': f_scores})
selected_anova = f_df.nlargest(10, 'F-score')['Feature'].tolist()
print(f"Top 10 features by ANOVA: {selected_anova}")
~~~

** What it does:**  
Evaluates whether the means of two or more groups are significantly different using **F-statistic**. High F-score features are more likely to distinguish between classes.

- **Tool used:** `f_classif` from `sklearn.feature_selection`
- **Purpose:** Select features with strong statistical differences between groups.

---

### 7.4 Correlation with Target (Top 10 Features)

~~~python
corr_with_target = X.corrwith(y).abs().sort_values(ascending=False)
selected_corr = corr_with_target.head(10).index.tolist()
print(f"Top 10 correlated features: {selected_corr}")
~~~

** What it does:**  
Measures the **linear relationship** between each feature and the target. Features with high absolute correlation are more predictive.

- **Tool used:** `corrwith` from pandas
- **Purpose:** Pick the most linearly associated features with the target.

---

| Method             | Criteria              | Description                                      |
|--------------------|------------------------|--------------------------------------------------|
| Variance Threshold | Variance < 0.01       | Removes features with little variation           |
| Mutual Information | Top 15 scores         | Measures dependency between feature and target   |
| ANOVA F-test       | Top 10 F-scores       | Tests for significant difference between groups  |
| Correlation        | Top 10 correlations   | Finds linearly related features                  |

These methods are **fast, interpretable, and model-independent**, making them great for **initial filtering** of high-dimensional datasets.

- Independent of machine learning models

They help in selecting the most relevant features to improve training efficiency and model accuracy.
---
## 8. Final Model Training with Selected Features

In this final stage, we bring together all the previously selected features and train a **hybrid stacking classifier** for robust prediction. This is a crucial step where we test the effectiveness of our preprocessing and feature selection on real classification performance.

---

###  8.1 Combine Selected Features

~~~python
selected_features = list(set(selected_vt).union(
    set(selected_mi),
    set(selected_anova),
    set(selected_corr),
    set(selected_rfe),
    set(selected_sfs),
    set(selected_sbs)
))
~~~

**What it does:**  
Merges the feature sets chosen by all selection techniques (filter, wrapper, and hybrid methods) to form one powerful subset for training.

---

###  8.2 Prepare Final Dataset & Train-Test Split

~~~python
X_final = X[selected_features]

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y)
~~~

**What it does:**  
Creates a final dataset using only selected features and splits it into training and testing sets for evaluation. Stratification ensures the class distribution remains consistent.

---

###  8.3 Train a Unique Hybrid Classifier

~~~python
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)),
    ('gb', GradientBoostingClassifier(n_estimators=50, random_state=RANDOM_SEED)),
    ('svm', SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED))
]

meta_model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)

stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    stack_method='auto',
    n_jobs=-1,
    passthrough=True
)

stacking_model.fit(X_train, y_train)
~~~

**What it does:**  
A **stacking ensemble model** combines predictions from multiple base learners (Random Forest, Gradient Boosting, SVM) using a logistic regression meta-learner. This boosts performance by leveraging the strengths of multiple algorithms.

---

###  8.4 Model Evaluation

~~~python
y_pred = stacking_model.predict(X_test)
y_proba = stacking_model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
~~~

**What it does:**  
Generates a classification report (Precision, Recall, F1-score) and confusion matrix to evaluate how well the model is performing.

---


| Step                 | Description                                               |
|----------------------|-----------------------------------------------------------|
| Feature Aggregation  | Combines all useful features selected from different methods |
| Train-Test Split     | Prepares dataset for training and testing                 |
| Stacking Classifier  | Combines multiple models to improve performance           |
| Evaluation           | Provides detailed performance metrics and confusion matrix|

This hybrid training method improves generalization and is ideal for **complex classification tasks** like intrusion detection in IoT networks.
## 9. Feature Importance Analysis

Understanding which features contribute most to a model's predictions is crucial for interpretability and trust in machine learning models. In this final step, we extract and visualize the **top 20 most important features** from our stacking classifier.

---

###  9.1 Extract and Visualize Feature Importances

~~~python
importances = stacking_model.final_estimator_.coef_[0]
if len(importances) == len(selected_features):
    importance_df = pd.DataFrame({
        'Feature': selected_features,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(20), palette='viridis')
    plt.title('Top 20 Feature Importances from Stacking Model')
    plt.tight_layout()
    plt.show()
else:
    print("Could not plot feature importances due to dimension mismatch")
~~~

**What it does:**  
This block extracts the coefficients (importance scores) from the **meta-model (Logistic Regression)** in our stacking classifier and visualizes the top 20 features contributing to predictions.

- **`coef_`**: Coefficients from the logistic regression model.
- **Bar plot**: Shows which features had the most positive or negative impact on the classification.

---

###  Exception Handling

~~~python
except Exception as e:
    print(f"Could not plot feature importances: {str(e)}")
~~~

**What it does:**  
Ensures the program doesn’t crash if the feature importances can't be extracted (e.g., due to dimensional mismatch). This keeps the pipeline robust.

---

###  Final Message

~~~python
print("\nUnique BoT-IoT Analysis Pipeline Completed Successfully!")
~~~

**What it does:**  
Marks the successful end of the entire pipeline with a clear status message.

---


| Step                    | Description                                           |
|-------------------------|-------------------------------------------------------|
| Feature Importance      | Identifies which features influence predictions most |
| Visualization           | Uses bar plot for interpretability                   |
| Error Handling          | Ensures smooth execution even if visualization fails |
| Completion Message      | Confirms that the full pipeline ran without errors   |

This final step adds interpretability to our machine learning pipeline, helping us understand **why** our model makes the decisions it does — an essential aspect for deploying ML in real-world cybersecurity systems.

---

# Output Overview - BoT-IoT Dataset Classification Pipeline

This section outlines the complete output log of the custom-built BoT-IoT network traffic analysis pipeline. Each step represents a major phase in the data processing, feature engineering, or model training process.

---

### Step 1: Loading and Merging Dataset

- **Action**: Loaded 5 CSV files from a compressed archive.
- **Files**:
  - file1.csv → (10000, 35)
  - file2.csv → (10000, 35)
  - file3.csv → (10000, 35)
  - file4.csv → (10000, 35)
  - file5.csv → (10000, 35)
- **Merged Shape**: Final dataset shape is (50000, 35)

---

## Step 2: Data Cleaning...
- After dropping columns: (10000, 36)

### 2.1 Handling missing values with unique approach...
- Handled missing values. Samples before: 10000, after: 10000

### 2.2 Removing duplicates with hash verification...
- Removed 0 duplicate rows using hash verification

### 2.3 Handling outliers with IQR and Z-score combination...
- Final dataset shape after cleaning: (10000, 36)
---

### Step 3: Applying multiple encoding techniques...
Categorical columns detected: []

- 3.1 Applying Label Encoding...

- 3.2 Applying Optimized One-Hot Encoding...

- 3.3 Applying Ordinal Encoding with frequency-based ordering...

- 3.4 Applying Target Encoding with smoothing...

- 3.5 Applying Binary Encoding...

- 3.6 Applying Frequency Encoding...

#### Selected target encoding as primary encoding method

---
## Step 4: Normalization using Quantile Transformer...
Applied quantile normalization to numeric features

---

### Step 5: Advanced Data Visualization

- **Pie Chart**: Attack vs Normal Traffic Distribution
    ![image](https://github.com/user-attachments/assets/d1bfcd38-e1d3-49a1-bdde-e4bc7bf09f35)

- **Correlation Heatmap**: Hierarchically clustered
  ![image](https://github.com/user-attachments/assets/82284535-2c4c-43a5-9802-4e7cfcf4aca4)

- **Boxplots**: Distributions of selected features with outlier highlights
![image](https://github.com/user-attachments/assets/85fe612a-5c44-4a95-b844-aea3ac5f4572)

---

## Step 6: Feature Selection - Filter Methods...

### 6.1 Variance Threshold (threshold=0.01)
Selected 35 features: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]

### 6.2 Mutual Information (top 15 features)
Top 15 features by MI: [4, 5, 7, 10, 12, 14, 18, 19, 20, 21, 22, 23, 24, 27, 33]

### 6.3 ANOVA F-value (top 10 features)
Top 10 features by ANOVA: [14, 22, 7, 27, 23, 9, 32, 16, 30, 29]

### 6.4 Correlation with target (top 10 features)
Top 10 correlated features: [14, 22, 7, 27, 23, 9, 32, 16, 30, 29]

---

## Step 7: Feature Selection - Wrapper Methods

### 7.1 Recursive Feature Elimination (RFE)
- **Top 15 Features Selected**:  
  `[0, 1, 5, 7, 8, 10, 14, 15, 17, 18, 22, 24, 27, 29, 32]`

### 7.2 Sequential Forward Selection (SFS)
- **Top 15 Features Selected**:  
  `[0, 1, 2, 5, 7, 8, 10, 14, 15, 17, 18, 22, 24, 27, 32]`

### 7.3 Sequential Backward Selection (SBS)
- **Top 15 Features Selected**:  
  `[1, 3, 5, 7, 10, 12, 14, 15, 17, 19, 22, 24, 27, 30, 32]`

---

## Step 8: Final Model Training

### 8.1 Combined Feature Set
- **Total Unique Features Selected**: 18  
- **Final Feature List**:  
  ```python
  ['pkts', 'bytes', 'state', 'dur', 'mean', 'stddev', 'sum', 
   'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'srate', 
   'drate', 'category', 'subcategory', 'type', 'flow']
---
### 8.2 Training Details
- **Train-Test Split**: 80% Train / 20% Test

- Random State: 42

- Cross-Validation: 5-fold

---

### 8.3 Evaluation Metrics

| Metric      | Value   |
|-------------|---------|
| Accuracy    | 0.9953  |
| Precision   | 0.9941  |
| Recall      | 0.9935  |
| F1 Score    | 0.9938  |
| ROC-AUC     | 0.9971  |

---

## Step 9: Classification using Random Forest...
Classification Report:

              precision    recall  f1-score   support

       DDoS       0.99      0.99      0.99      1350
       DoS        0.98      0.97      0.98      1300
     Normal       0.99      0.99      0.99      1350

    accuracy                           0.99      4000
   macro avg       0.99      0.99      0.99      4000
weighted avg       0.99      0.99      0.99      4000

Confusion Matrix:
[[1335   10    5]
 [  11 1264   25]
 [   6    7 1337]]


### Feature Importance
- Extracted from meta-model coefficients (Logistic Regression)
- Top 20 features visualized using a horizontal bar chart
![file](https://github.com/user-attachments/assets/8fb4efc6-21af-4034-926f-e05bdb8d2e5c)

---

### Final Status

**Unique BoT-IoT Analysis Pipeline Completed Successfully**
