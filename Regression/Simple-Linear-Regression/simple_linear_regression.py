# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 20:07:09 2018

@author: Vaibhan
"""

import numpy as np
import matplotlib.pyplot as mat
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:, 0].values
Y = dataset.iloc[:, 1].values
print (X)
print (Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 1/3,random_state = 0)

#feature scaling not required
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train.reshape(-1, 1), Y_train.reshape(-1, 1))

y_pred = regressor.predict(X_test.reshape(-1, 1))

mat.scatter(X_train, Y_train, color = 'red')
mat.plot(X_train, regressor.predict(X_train.reshape(-1, 1)), color = 'blue')
mat.title('Salary vs Experience (Training Set)')
mat.xlabel('Years of Experience')
mat.ylabel('Salary')
mat.show()

mat.scatter(X_test, Y_test, color = 'red')
mat.plot(X_train, regressor.predict(X_train.reshape(-1 ,1)),color = 'blue')
mat.title('Salary vs Experience (Training Set)')
mat.xlabel('Years of Experience')
mat.ylabel('Salary')
mat.show()



