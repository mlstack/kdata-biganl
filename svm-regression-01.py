#!/usr/bin/env /c/Apps/Anaconda3/python

"""
SVM - Regression
Program Code Name : svm-regression-01.py
Author : 이이백(yibeck.lee@gmail.com)
"""

print(__doc__)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR



from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

np.set_printoptions(linewidth=400)    # screen size

np.set_printoptions(threshold=np.inf) # print all numpy output

dfTrain = pd.read_csv("C://Home//data//file//data01-train.csv",header=None)

dfTest = pd.read_csv("C://Home//data//file//data01-test.csv",header=None)

# dfTest = pd.read_csv("C://Home//GitHub//deepbig-ng2-login//resources//data//data01-train.csv",header=None)

print(dfTrain[0:10])

arrTrain = dfTrain.as_matrix().astype(int)

print(arrTrain[0:10])

print(arrTrain.shape)



# print(name_dtype)

arrTrain = np.array(arrTrain)

print(arrTrain.dtype)



print(np.unique(arrTrain[:,0])) # binary label 

numClass = np.unique(arrTrain[:,1]).size

print('numClass',numClass)

labelTrain = arrTrain[:,0].astype(int) 

# print(labelTrain[0:100])

# print('[labelTrain[0,1] : ', np.bincount(labelTrain)) # get frequency of [0,1]

featuresTrain = arrTrain[:,2:]







svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

svr_lin = SVR(kernel='linear', C=1e3)

svr_poly = SVR(kernel='poly', C=1e3, degree=2)



pred_rbf = svr_rbf.fit(featuresTrain, labelTrain).predict(featuresTrain)

pred_lin = svr_lin.fit(featuresTrain, labelTrain).predict(featuresTrain)

pred_poly = svr_poly.fit(featuresTrain,featuresTrain).predict(featuresTrain)



"""

arrTest = dfTest.as_matrix().astype(int)

predTest = arrTest[:,1:]

featuresTest = arrTest[:,2:]

predLabelTest = svc.predict(featuresTest)

print(predLabelTest)

lw = 2

plt.scatter(featuresTrain, featuresTrain, color='darkorange', label='data')

plt.plot(featuresTrain, pred_rbf, color='navy', lw=lw, label='RBF model')

plt.plot(featuresTrain, pred_lin, color='c', lw=lw, label='Linear model')

plt.plot(featuresTrain, pred_poly, color='cornflowerblue', lw=lw, label='Polynomial model')

plt.xlabel('data')

plt.ylabel('target')

plt.title('Support Vector Regression')

plt.legend()

plt.show()

"""