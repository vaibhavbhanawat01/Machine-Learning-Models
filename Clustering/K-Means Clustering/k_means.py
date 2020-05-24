# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 19:31:21 2019

@author: Vaibhan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as mat

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values;

from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
mat.plot(range(1, 11), wcss)
mat.title('The elbow method')
mat.xlabel('No of cluster')
mat.ylabel('WCSS')
mat.show()

kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
kmeans_predict = kmeans.fit_predict(X)

mat.scatter(X[kmeans_predict == 0, 0], X[kmeans_predict == 0, 1], s = 100, c = 'red', label = 'Careful')
mat.scatter(X[kmeans_predict == 1, 0], X[kmeans_predict == 1, 1], s = 100, c = 'blue', label = 'Standard')
mat.scatter(X[kmeans_predict == 2, 0], X[kmeans_predict == 2, 1], s = 100, c = 'green', label = 'Target')
mat.scatter(X[kmeans_predict == 3, 0], X[kmeans_predict == 3, 1], s = 100, c = 'cyan', label = 'Careless')
mat.scatter(X[kmeans_predict == 4, 0], X[kmeans_predict == 4, 1], s = 100, c = 'magenta', label = 'Sensible')
mat.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')

mat.title('CLuster of clients')
mat.xlabel('Anula Income (k)$')
mat.ylabel('Spending Score (1-100)')
mat.legend()
mat.show()

