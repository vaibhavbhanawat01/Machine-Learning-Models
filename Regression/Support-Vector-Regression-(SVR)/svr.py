# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 12:23:04 2018

@author: Vaibhan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values;
Y = dataset.iloc[:, 2:3].values;

#need to do feature scaling since SVM donesn't do feature scaling by default

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)

#regressor for SVR
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, Y)

y_pred = sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([6.5]).reshape(-1, 1))))
print (y_pred)

plt.scatter(X,Y,color = 'red')
plt.plot(X, regressor.predict(X),color = 'blue')
plt.title('Truth or Bluff(SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()