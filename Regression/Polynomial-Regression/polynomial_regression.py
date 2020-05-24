# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 12:22:41 2018

@author: Vaibhan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2:3].values

#fillting in linear

from  sklearn.linear_model import LinearRegression
lReg = LinearRegression()
lReg.fit(X, Y)

#fitting in polynimals

from sklearn.preprocessing import PolynomialFeatures
pReg = PolynomialFeatures(degree = 4)
X_poly = pReg.fit_transform(X)
lReg2 = LinearRegression()
lReg2.fit(X_poly, Y)


#visual of linear 
plt.scatter(X, Y, color = 'red')
plt.plot(X, lReg.predict(X), color = 'black')
plt.title('Position level/Salary')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#visual of Polynomial 
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, Y,color = 'blue')
plt.plot(X_grid, lReg2.predict(pReg.fit_transform(X_grid)), color = 'Black')
plt.title('Position Level/Salary')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#predict new result by linear

lReg.predict(6.5)
#predict new result by polynomial

lReg2.predict(pReg.fit_transform(6.5))

