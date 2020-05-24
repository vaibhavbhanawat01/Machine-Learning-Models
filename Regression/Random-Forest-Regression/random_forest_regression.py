# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 23:44:04 2018

@author: Vaibhan
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2:3].values

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 200, random_state = 0)
regressor.fit(X, Y)
Y_pred = regressor.predict(np.array(6.5).reshape(1, -1))

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Position/Salary (Random forest regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()