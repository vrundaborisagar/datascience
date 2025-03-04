# EXP 4: Data preprocessing- Normalization  
Normalization Techniques are essential in data preprocessing to scale numerical features to a common range, reducing the impact of different units and improving model performance in machine learning by ensuring fair comparisons between variables. 
## Introduction:
Normalization is a scaling technique in Machine Learning applied during data preparation to change the values of numeric columns in the dataset to use a common scale.
<br>
The basic formula for Normalization can be given as below:
<br>

Normalization (Min-Max Scaling):
<br>

- Formula:<br>

     ![image](https://github.com/user-attachments/assets/70fc8363-9bee-49e5-9066-608c82fe4fc6)

<br>

- Purpose: Rescales features to a specific range, typically [0, 1].<br>

- Impact: Changes the range of the data but preserves the distribution's shape.

## Why Min-Max Scaling?
Min-Max Scaling transforms data to a fixed range, usually [0,1] or [-1,1], making it useful for preserving relationships between data points while preventing large values from dominating smaller ones.
## Key Points:

- Prevents features from dominating: Min-Max Scaling ensures that features with larger values (e.g., salary) don’t overshadow smaller ones (e.g., rating). This helps maintain equal importance among features, leading to balanced model learning and better performance.
- Improves Convergence Speed in ML algorithms: Min-Max Scaling ensures all features are on a similar scale, preventing larger-magnitude features (e.g., salary) from dominating smaller ones (e.g., age). This results in faster and more stable training, especially for Neural Networks, SVMs, and Gradient Descent-based models.
- Essential for Distance-Based Models: Min-Max Scaling is crucial for KNN, K-Means, and other distance-based models, as features with larger values can dominate distance calculations, leading to biased results. Scaling ensures fair comparisons and accurate clustering or classification.
- Improves Neural Network Efficiency: Min-Max Scaling ensures inputs stay within small ranges (e.g., [0,1]), helping activation functions like Sigmoid and Tanh perform optimally, leading to better learning and faster convergence.

## Example of Normalization(Min-Max Scaling) In Python
```
from sklearn.preprocessing import MinMaxScaler
import numpy as np
data= np.array([[100],[200],[300],[400],[500]])
scaler= MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(data)
print(scaled_data)

```
## Comparison with Other Scaling Techniques
| Feature          | Description                                       | Example       |
|-----------------|---------------------------------------------------|--------------|
| **Min-Max**     | Scales between `0` and `1`                        | `[0, 1]`     |
| **Z-Score**     | Centers around **mean**, unit variance            | `(-∞, ∞)`    |
| **Robust**      | Uses *median* and **IQR**, robust to outliers      | `[Q1, Q3]`   |


