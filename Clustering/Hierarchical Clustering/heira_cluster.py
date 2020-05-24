# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 20:00:17 2019

@author: Vaibhan
"""

import pandas as pd
import matplotlib.pyplot as mat
import numpy as np

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

#find out optimal no of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))

mat.title('Dendrogram')
mat.xlabel('Customers')
mat.ylabel('Euclidean Distance')
mat.show()

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
hc_pred = hc.fit_predict(X)

mat.scatter(X[hc_pred == 0, 0], X[hc_pred == 0, 1], s = 100, c = 'red', label = 'Careful')
mat.scatter(X[hc_pred == 1, 0], X[hc_pred == 1, 1], s = 100, c = 'blue', label = 'Standard')
mat.scatter(X[hc_pred == 2, 0], X[hc_pred == 2, 1], s = 100, c = 'green', label = 'Target')
mat.scatter(X[hc_pred == 3, 0], X[hc_pred == 3, 1], s = 100, c = 'cyan', label = 'Careless')
mat.scatter(X[hc_pred == 4, 0], X[hc_pred == 4, 1], s = 100, c = 'magenta', label = 'Sensible')

mat.title('CLuster of clients')
mat.xlabel('Anula Income (k)$')
mat.ylabel('Spending Score (1-100)')
mat.legend()
mat.show()