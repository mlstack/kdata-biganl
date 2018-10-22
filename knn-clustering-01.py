#!/usr/bin/env /c/Apps/Anaconda3/python

"""
knn Clustering
Program Code Name : knn-clustering-01.py
Author : 이이백(yibeck.lee@gmail.com)
"""

print(__doc__)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.metrics import mean_squared_error

np.set_printoptions(linewidth=400)    # screen size
np.set_printoptions(threshold=np.inf) # print all numpy output
dfTrain = pd.read_csv("C://Home//data//file//data01-train.csv",header=None)
dfTest = pd.read_csv("C://Home//data//file//data01-test.csv",header=None)
# dfTest = pd.read_csv("C://Home//GitHub//deepbig-ng2-login//resources//data//data01-train.csv",header=None)
# print(dfTrain[0:10])
# print(dfTrain.columns.values)
# print(dfTrain[0:2][[1,2,12]])
dfTrain.columns = ['Y1','Y2','X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']
# print(dfTrain.columns.values)
dfTest.columns = ['Y1','Y2','X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']
# print(dfTest.columns[[2,3]])
model = neighbors.NearestNeighbors().fit(dfTrain[['X1','X2']])
print(model)
distances, indices = model.kneighbors(dfTest[['X1','X2']])
print(indices[0:100])
print(distances[0:100])


print(model.kneighbors_graph(dfTest[['X1','X2']]).toarray())
