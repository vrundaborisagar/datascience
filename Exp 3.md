# EXP 3: Data Preprocessing- Applying encoding techniques. 
<br>
Encoding is an essential step in data preprocessing that converts categorical data into numerical form, making it usable for machine learning models. This repository explains different encoding techniques and how to use them effectively.  

<br>
 
##  Introduction  
Machine learning models require numerical data for training. However, datasets often contain categorical variables that need to be converted into numbers. This is where **encoding techniques** help.  

 

##  Types of Encoding  

<br>
1. Label Encoding.
<br>
2. One-hot Encoding.
<br>
3. Ordinal Encoding.
<br>
4. Target Encoding. 
<br>
5. Binary Encoding.
<br>
6. Frequency Encoding.
 

### 1️ Label Encoding  

Each category is given a unique number (e.g., Low= 0, High = 1, Medium = 2).
     


```
from sklearn.preprocessing import LabelEncoder
data=['Low','High','Medium','High','Medium']
encoder= LabelEncoder()
encoded_data= encoder.fit_transform(data)
print(f"Label encoded data: {encoded_data}")
```
 
### 2. One hot encoding:
 Creates separate columns for each category and marks them as 1 (present) or 0 (not present).

```
import pandas as pd
data=['Red','Blue','Green','Blue','Red']
df= pd.DataFrame(data,columns=['Color'])
one_hot_encoded=pd.get_dummies(df['Color'])
print("one hot encoded: \n")
print(one_hot_encoded)
```
 
### 3. Ordinal encoding:
 Assigns numbers to categories based on their order (e.g., Low= 0, Medium = 1, High = 2).
<br>

``` 
from sklearn.preprocessing import OrdinalEncoder
data=[['Low'],['High'],['Medium'],['High'],['Medium']]
encoder= OrdinalEncoder(categories=[['Low','Medium','High']])
encoded_data=encoder.fit_transform(data)
print(f"Ordinal Encoded Data: {encoded_data}")
```
 
### 4. Target encoding:
Replaces categories with the average value of the target variable for that category.
<br>

```
!pip install category_encoders
import pandas as pd
import category_encoders as ce
data= {'Color':['Red','Blue','Green','Blue','Red','Blue','Green','Green','Green','Blue'],'Target':['1','0','0','1','1','1','0','1','0','1']}
df=pd.DataFrame(data)
df['Target'] = df['Target'].astype(int)
encoder= ce.TargetEncoder(cols=['Color'])
encoded_data= encoder.fit_transform(df['Color'],df['Target'])
print(f"Target encoded: {encoded_data}")
```
 
### 5. Binary encoding:
 Converts category values into binary (0s and 1s) and stores them in multiple columns.
<br>

```
import category_encoders as ce
data=['Red','Green','Blue','Red','Grey']
encoder = ce.BinaryEncoder(cols=['Color'])
encoded_data= encoder.fit_transform(pd.DataFrame(data,columns=['Color']))
print("binary encoded: \n")
print(encoded_data)
```
 
### 6. Frequency encoding:
Replaces categories with how often they appear in the dataset.
<br>

```
import pandas as pd
data=['Red','Green','Blue','Red','Red']
series_data= pd.Series(data)
frequency_encoding= series_data.value_counts()
encoded_data= [frequency_encoding[x] for x in data]
print("frequency encoded: \n")
print("encoded data: ",encoded_data)
```


