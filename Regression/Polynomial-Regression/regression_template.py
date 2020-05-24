# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 22:15:48 2018

@author: Vaibhan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values
#
#from sklearn.cross_validation import train_test_split
#X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 0)



y_pred  = regressor.predict(X)

plt.scatter(X,Y,color = 'red')
plt.plot(X,regressor.predict(X),color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

X_grid = np.arange(min(X),max(Y),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,Y,color = 'red')
plt.plot(X,regressor.predict(X_grid),color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
