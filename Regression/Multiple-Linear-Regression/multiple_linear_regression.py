# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 18:59:43 2018

@author: Vaibhan
"""

import numpy as np
import matplotlib as plt
import pandas as pd

# using using p value and adjusted r squared
#def backwardElimination(x, SL):
#    numVars = len(x[0])
#    temp = np.zeros((50,6)).astype(int)
#    for i in range(0, numVars):
#        regressor_OLS = sm.OLS(y, x).fit()
#        maxVar = max(regressor_OLS.pvalues).astype(float)
#        adjR_before = regressor_OLS.rsquared_adj.astype(float)
#        if maxVar > SL:
#            for j in range(0, numVars - i):
#                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
#                    temp[:,j] = x[:, j]
#                    x = np.delete(x, j, 1)
#                    tmp_regressor = sm.OLS(y, x).fit()
#                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
#                    if (adjR_before >= adjR_after):
#                        x_rollback = np.hstack((x, temp[:,[0,j]]))
#                        x_rollback = np.delete(x_rollback, j, 1)
#                        print (regressor_OLS.summary())
#                        return x_rollback
#                    else:
#                        continue
#    regressor_OLS.summary()
#    return x

#Using p value only
def backwardElimination(x, sl, y): 
    numVars = len(x[0]) 
    print (numVars)
  
    for i in range (0, numVars):
        regressorOLS = sm.OLS(y, x).fit()
        maxVar = max(regressorOLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0 , numVars - i):
                if (regressorOLS.pvalues[j].astype(float)==maxVar):
                    x = np.delete(x, j, 1)
    regressorOLS.summary()
    return x;

        

dataset = pd.read_csv("50_Startups.csv")
print (dataset)
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,4].values

print (X)
print (Y)


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder = LabelEncoder();
X[:,3] = labelEncoder.fit_transform(X[:,3])
onehotEncoder = OneHotEncoder(categorical_features  = [3])
X = onehotEncoder.fit_transform(X).toarray()

print (X)

X = X[:,1:]

from sklearn.model_selection import train_test_split
X_train, X_test,Y_train,Y_test = train_test_split(X,Y,  test_size = 0.2 , random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test);

#using backward elimination
import statsmodels.regression.linear_model as sm
X = np.append(arr = np.ones((50,1)).astype(int),values = X,axis=1)

X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = Y,exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = Y,exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog = Y,exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = Y,exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog = Y,exog = X_opt).fit()
regressor_OLS.summary()



X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt,0.05,Y)
#print (X_Modeled)

