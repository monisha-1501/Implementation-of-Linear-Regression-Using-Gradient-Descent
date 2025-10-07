# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
Developed by: Monisha D
RegisterNumber:  25007487
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = {
    'Hours': [1.1, 2.5, 3.2, 4.5, 5.1, 6.7, 8.0, 2.8, 3.5, 1.5, 9.2, 5.5, 3.3, 2.2, 7.7, 4.8, 6.9, 7.4, 3.9, 1.9, 8.5, 2.7, 5.9, 4.3, 9.7],
    'Scores': [22, 45, 50, 54, 60, 65, 78, 48, 52, 30, 88, 66, 55, 40, 85, 62, 75, 80, 58, 38, 82, 50, 72, 54, 95]
}

df = pd.DataFrame(data)
X = df['Hours'].values
Y = df['Scores'].values

X = X.reshape(-1, 1)

X = X / np.max(X)
Y = Y / np.max(Y)

m = 0
c = 0
learning_rate = 0.01
epochs = 1000
n = len(X)

for i in range(epochs):
    Y_pred = m * X + c
    error = Y_pred - Y

    dm = (1/n) * np.sum(error * X)
    dc = (1/n) * np.sum(error)

    m -= learning_rate * dm
    c -= learning_rate * dm

print("Slope (m):", m)
print("Intercept (c):", c)

Y_final_pred = m * X + c

plt.scatter(X, Y, color='blue', label='Actual Data')
plt.plot(X, Y_final_pred, color='red', label='Regression Line')
plt.title('Linear Regression using Gradient Descent')
plt.xlabel('Hours (Normalized)')
plt.ylabel('Scores (Normalized)')
plt.legend()
plt.show()

mse = np.mean((Y - Y_final_pred) ** 2)
print("Mean Squared Error:", mse)

```

## Output:
Slope (m): 0.3873420952634226
Intercept (c): 0.3873420952634226
<img width="774" height="585" alt="{8EE28980-B208-4221-8F6C-C0B6BF5A8F21}" src="https://github.com/user-attachments/assets/16884660-0111-4938-b7e4-d97603801c28" />
Mean Squared Error: 0.04874960390659464



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
