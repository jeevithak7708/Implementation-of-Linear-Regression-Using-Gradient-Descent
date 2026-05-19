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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("50_Startups.csv")
X=data['R&D Spend'].values
Y=data['Profit'].values
X=(X-X.mean())/X.std()
m=0
b=0
lenr=0.01
epo=1000
n = len(X)
for i in range(epo):
    Y_pred=(m*X)+b
    dm=(-2/n)*np.sum(X*(Y-Y_pred))
    db=(-2/n)*np.sum(Y-Y_pred)
    m=m-lenr*dm
    b=b-lenr*db
print("Slope:",m)
print("Intercept:",b)
Y_pred=m*X+b
plt.scatter(X,Y,label="Actual Data")
plt.plot(X,Y_pred,label="Best fitted")
plt.xlabel("R&D Spend")
plt.ylabel("Profit")
plt.title("Gradient descent")
plt.legend()
plt.show()

```

## Output:
<img width="987" height="759" alt="Screenshot 2026-05-19 183548" src="https://github.com/user-attachments/assets/597776eb-60c4-42a8-85ed-bc364c4f2d09" />




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
