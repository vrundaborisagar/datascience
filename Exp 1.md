# EXP 1: Loading and Merging Datasets in Python
<br>
This repository contains the code and results for the first experiment in the Data Science Fundamentals with Python course. The objective of this experiment is to load and merge datasets obtained from the UCI Machine Learning Repository.



## Steps:

<br>
Loading Datasets: The datasets are loaded using the Pandas library from the UCI ML Repository.
<br>
Inspecting Datasets: The first few rows of each dataset are displayed to understand their structure.
<br>
Merging Datasets: The datasets are merged on a common column (e.g., 'ID').
<br>
Saving the Merged Dataset: The merged dataset is optionally saved as a CSV file.


## Concepts Used-
<br>
Pandas Library: Used for data manipulation and analysis.
  <br>
DataFrames: Data structures provided by Pandas to store and manipulate tabular data.
<br>
Merging Datasets: The pd.merge() function is used to combine two datasets based on a common key.


## Steps to Reproduce

###  1.Import necessary library
<br>

Pandas is a powerful Python library for data manipulation, analysis, and processing using DataFrames and Series.
```
import pandas as pd
```
###  2.Load the datasets 
<br>
Here, I've taken the wine dataset  

```
df1= pd.read_csv('/content/drive/MyDrive/winequality-red.csv', delimiter=';')
df2= pd.read_csv('/content/drive/MyDrive/winequality-white.csv',delimiter=';')
```

### 3. Inspect the datasets 
<br>
Show the first few rows of each dataset to verify they loaded correctly.

```
print("red wine data: ")
df1['type']='red'
print(df1.head())
print("white wine data: ")
df2['type']='white'
print(df2.head())
```

### 4.Merge the Datasets
Merge the datasets on a common column (e.g., 'ID').
```
merged= pd.concat([df1,df2],ignore_index= True)
```
### 5. Inspect the merged Dataset
<br>
Display the first few rows of the merged dataset.

```
print("combined: ")
print(merged.head())
```
