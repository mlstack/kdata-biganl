#!/usr/bin/env /c/Apps/Anaconda3/python

"""
k-means clustering
Author : 이이백(yibeck.lee@gmail.com)
modify source from
# Code source: Gael Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause
"""
print(__doc__)

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# X = np.array([[1, 2], [1, 4], [1, 0],
#               [4, 2], [4, 4], [4, 0]])
# kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
# print(kmeans.labels_)
# pred= kmeans.predict([[0, 0], [4, 4]])
# print(pred)
# print(kmeans.cluster_centers_)


np.set_printoptions(linewidth=400)    # screen size
np.set_printoptions(threshold=np.inf) # print all numpy output
dfTrain = pd.read_csv("C://Home//data//file//data01-train.csv",header=None)
dfTest = pd.read_csv("C://Home//data//file//data01-test.csv",header=None)

dfTrain = dfTrain[0:1000]
print(dfTrain.columns.values)
print(dfTrain[0:2][[1,2,12]])
dfTrain.columns = ['Y1','Y2','V1','V2','V3','V4','V5','V6','v7','V8','V9','V10','V11']
dfTest.columns = ['Y1','Y2','V1','V2','V3','V4','V5','V6','v7','V8','V9','V10','V11']
print(dfTrain.columns.values)

kmeans = KMeans(n_clusters=3, random_state=0).fit(dfTrain[['V9','V10','V11']])
print(kmeans)
print(kmeans.cluster_centers_)

# for i in range(kmeans.labels_.size):
	# print(kmeans.labels_[i])

pred=kmeans.predict(dfTest[['V9','V10','V11']])
# for i in range(pred.size):
# 	print(pred[i])

X = dfTrain[['V4','V5','V6']].as_matrix().astype(int)
y = dfTrain[['Y2']]
estimators = [('k_means_2', KMeans(n_clusters=2)),
              ('k_means_3', KMeans(n_clusters=3))]

print(estimators)
fignum = 1
titles = ['2 clusters', '3 clusters']
for name, est in estimators:
    fig = plt.figure(fignum, figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    est.fit(X)
    labels = est.labels_
    print(labels[0:3])
    ax.scatter(X[:, 0], X[:, 1], X[:, 2],
           c=labels.astype(np.float), edgecolor='k')

    ax.set_xlabel('V4')
    ax.set_ylabel('V5')
    ax.set_zlabel('V6')
    ax.set_title(titles[fignum - 1])
    fignum = fignum + 1


plt.show()