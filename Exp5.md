# Experiment 5- Plotting Techniques 
## 1. Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split# 1. Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

## 2. Read the dataset 
df = pd.read_csv('/content/drive/MyDrive/iris.data', header=None)
print(df.head())

### Assign column names (dataset doesnt have header)
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

## 3. Checking for null and NaN values
print("\nChecking for missing values:")
print(df.isnull().sum())

## 4. Plot all graphs (histograms and pairplots)
plt.figure(figsize=(12, 6))
df.hist(bins=20, figsize=(10, 8), grid=False)
plt.suptitle("Histograms of Features")
plt.show()

sns.pairplot(df, hue="species")
plt.show()

## 5.  plot for outlier detection
plt.figure(figsize=(10, 6))
df.drop(columns=['species']).boxplot()
plt.title("Boxplot of Numerical Features")
plt.show()

## 6. Removing outliers using IQR method
Q1 = df.drop(columns=['species']).quantile(0.25)
Q3 = df.drop(columns=['species']).quantile(0.75)
IQR = Q3 - Q1

### Defining the range for outlier detection
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

### Filtering outliers
df_cleaned = df[~((df.drop(columns=['species']) < lower_bound) | (df.drop(columns=['species']) > upper_bound)).any(axis=1)]

print("\nDataset shape before outlier removal:", df.shape)
print("Dataset shape after outlier removal:", df_cleaned.shape)

## 7. Train-Test Split
X = df_cleaned.drop(columns=['species'])  
y = df_cleaned['species'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nTraining set shape:", X_train.shape)
print("Test set shape:", X_test.shape)
