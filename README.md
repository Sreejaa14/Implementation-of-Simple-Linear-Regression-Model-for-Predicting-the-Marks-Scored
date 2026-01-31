# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Input the dataset with hours studied and marks scored.

2. Split the data into training and testing sets.

3. Train the Simple Linear Regression model.

4. Predict marks and evaluate the results. 

## Program:
```
/*
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: R.SREEJAA
RegisterNumber: 212225220101
*/
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dataset
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
Y = np.array([40, 50, 60, 70, 80])

# Model
model = LinearRegression()
model.fit(X, Y)

# Prediction
predicted_marks = model.predict([[6]])
print("Predicted marks:", predicted_marks[0])

# Graph
plt.scatter(X, Y)
plt.plot(X, model.predict(X))
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.show()

  
*/
```

## Output:

<img width="867" height="590" alt="Screenshot 2026-01-31 155337" src="https://github.com/user-attachments/assets/604449d9-872b-45ef-a0a8-3b4e7581392a" />

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
