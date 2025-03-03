# EXP 2: Data Preprocessing - Cleaning dataset
 Data Cleaning is a crucial step in data preprocessing that involves detecting and correcting errors, handling missing values, removing duplicates, and ensuring data consistency to improve the quality and reliability of datasets for analysis and machine learning.
 
## Steps:
<br>
1.Loading datasets: The datasets are loaded using the Pandas library from the UCI ML Repository.
<br>
2.Handling Missing Values:Handle missing values by removing them or replacing Null/NaN with defaults, statistics, or predictions.
<br>
3.Removing Duplicates: Remove duplicates by dropping them or keeping the first/last occurrence based on data needs.
<br>
4.Handling Outliers:Handle outliers by removing or capping values beyond the IQR range (1.5×IQR rule).

## Concepts Used:
<br>
Pandas Library: Used for data manipulation and analysis.
<br>
DataFrames: Data structures provided by Pandas to store and manipulate tabular data.
<br>
Data Cleaning: Techniques such as removing duplicates, handling missing values, and standardizing data.

## Steps to Reproduce: 

### 1.Import necessary library

Pandas is a powerful Python library for data manipulation, analysis, and processing using DataFrames and Series.
<br>


```
import pandas as pd
```

### 2.Load the Datasets
<br>

- Here,I've used the wine dataset
<br>

```
df1= pd.read_csv('/content/drive/MyDrive/winequality-red.csv', delimiter=';')
df2= pd.read_csv('/content/drive/MyDrive/winequality-white.csv',delimiter=';')
```

### 3.Check for Null values/ Missing values
<br>

- We can inspect for null values in 2 ways(using 2 functions) i.e. 
<br>

a)Isnull() 
<br>

```
print("checking for null values(is null)")
df1.isnull()

```
<br>
b)Notnull() 
<br>

```
print("checking for null values(not null)")
df1.notnull()
```
### 4.Handling NAN values
<br>
NAN stands for Not A Numeric Value i.e this will check for any value which is not a number/ a numeric value.

<br>
<br>

- There are two popular functions to handle NAN values i.e.,
   <br>
   <br>

   
   - dropna():Removes missing (NaN) values from a DataFrame or Series.
  <br>
  
  ```
    df.dropna(inplace=True)
  ```
  <br>
  
  - fillna():Replaces missing values with a specified value.
  ```
  df.fillna(130,inplace=True)
  ```

### 5.Handle Duplicate Rows:
<br>

- Identify and remove duplicate rows.

```
df.duplicated()
df.drop_duplicates(inplace=True)
df.duplicated()
```


### 6. Handling Outliers using IQR method
<br>

- The Interquartile Range (IQR) method is a statistical approach used to identify and remove extreme values from a dataset.
<br>
<br>

 -  The Sub-steps to handle outliers using IQR are-
    <br>
    <br>
    
1. Stripping Extra Spaces from Column Names:Removes extra spaces to avoid referencing issues.
    <br>
    
```
df.columns = df.columns.str.strip()
print("Column Names:", df.columns)

```
<br>
    2. Calculating Inter-Quartile Range(IQR)
    <br>

      q1 = df['RATING'].quantile(0.25)
      q3 = df['RATING'].quantile(0.75)
      iqr = q3 - q1
Where,
   <br>
    Q1 (25th percentile): The value below which 25% of the data falls.<br>
    Q3 (75th percentile): The value below which 75% of the data falls.<br>
    IQR: The range between Q1 and Q3, representing the middle 50% of the data.<br>
   <br>
   <br>
    3. Defining the Outlier Boundaries:The 1.5 × IQR rule defines outliers as values lying
    below Q1 - 1.5 × IQR or above Q3 + 1.5 × IQR.
    
     lower_bound = q1 - 1.5 * iqr
     upper_bound = q3 + 1.5 * iqr

  <br>
    4. Filtering Out Outliers:Removes outliers and saves the cleaned dataset.
    <br>
    
    
    df_clean = df[(df['RATING'] >=lower_bound) & (df['RATING'] <= upper_bound)]
    
    df_clean.to_csv("/content/drive/MyDrive/cleaned_movies.csv", index=False)
    print(" Outliers removed! Cleaned data saved as 'cleaned_movies.csv'.")
