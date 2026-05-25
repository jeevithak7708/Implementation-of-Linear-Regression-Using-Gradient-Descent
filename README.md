# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Initialize w, b, learning rate and dataset.

2.Normalize input and output values.

3.Compute prediction y=wx+b.

4.Update parameters using gradient descent.

5.Repeat and use final model for prediction.

## Program:
```
DEVELOPED BY : JEEVITHA K

REGISTER NO : 212225040149



# Import necessary modules
import numpy as np

# Sample dataset
# Features: [Hours Studied, Attendance, Previous Marks]
X = np.array([
    [2, 80, 50],
    [3, 60, 40],
    [5, 90, 70],
    [7, 85, 80],
    [9, 95, 90]], dtype=float)
# Target: Marks Scored
y = np.array([50, 45, 70, 80, 95], dtype=float)

# Feature normalization
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

# Add bias term (intercept)
X = np.c_[np.ones(X.shape[0]), X]  # shape becomes (n_samples, n_features + 1)

# Initialize weights
n_features = X.shape[1]
weights = np.zeros(n_features)

# Hyperparameters
learning_rate = 0.01
epochs = 1000

# Stochastic Gradient Descent
for epoch in range(epochs):
    for i in range(X.shape[0]):
        xi = X[i]
        yi = y[i]
        y_pred = np.dot(xi, weights)
        error = y_pred - yi
        # Update weights
        weights -= learning_rate * error * xi
print("Trained Weights (including intercept):", weights)

# Make predictions
y_pred_all = np.dot(X, weights)
print("Predicted values:", y_pred_all)


```

## Output:

<img width="1136" height="64" alt="Screenshot 2026-05-25 074519" src="https://github.com/user-attachments/assets/6306814c-2a13-4f41-8efb-3c5d3f2e9f0e" />




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
