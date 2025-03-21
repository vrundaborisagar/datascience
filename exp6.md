# Iris Dataset Visualization using Matplotlib and Seaborn
## Project Overview
This project demonstrates various data visualization techniques using the Iris dataset from the UCI Machine Learning Repository. The dataset consists of three species of Iris flowers (Setosa, Versicolor, Virginica) with four numerical features (sepal length, sepal width, petal length, petal width).

## Dataset Information
Dataset Name: iris.data
</br>
Source: UCI Machine Learning Repository
</br>
Format: CSV
</br>
Number of Instances: 150
</br>
Number of Attributes: 4 numerical + 1 categorical (species)
## Column Names
### Feature	Description
sepal_length = Sepal length in cm
</br>
sepal_width	= Sepal width in cm
</br>
petal_length	= Petal length in cm
</br>
petal_width =	Petal width in cm
## code for Advanced Visualizations

### import data
```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```
### Load the dataset
```
df = pd.read_csv('/content/drive/MyDrive/iris.data', header=None)
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
```

### Scatter plot - Sepal Length vs Sepal Width
```
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['sepal_length'], y=df['sepal_width'], hue=df['species'])
plt.title("Scatter Plot: Sepal Length vs Sepal Width")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.show()
```
### Bar plot - Average Sepal Width per Species
```
plt.figure(figsize=(8, 6))
sns.barplot(x=df['species'], y=df['sepal_width'], palette="viridis")
plt.title("Average Sepal Width per Species")
plt.xlabel("Species")
plt.ylabel("Average Sepal Width (cm)")
plt.show()
```
### Heatmap - Correlation between features
```
plt.figure(figsize=(8, 6))
sns.heatmap(df.drop(columns=['species']).corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()
```
### Output:

```
Sepal Length  Sepal Width  Petal Length  Petal Width      Species
0           5.1          3.5           1.4          0.2  Iris-setosa
1           4.9          3.0           1.4          0.2  Iris-setosa
2           4.7          3.2           1.3          0.2  Iris-setosa
3           4.6          3.1           1.5          0.2  Iris-setosa
4           5.0          3.6           1.4          0.2  Iris-setosa
```
![download (5)](https://github.com/user-attachments/assets/26204a3c-a667-4150-9a47-486dfe019b45)
![download (6)](https://github.com/user-attachments/assets/acbc78a4-933b-4ae0-a422-fd2257ff68c8)
