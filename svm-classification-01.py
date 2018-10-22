#!/usr/bin/env /c/Apps/Anaconda3/python

"""
SVM - Classification
Program Code Name : svm-classification-01.py
Author : 이이백(yibeck.lee@gmail.com)
"""
print(__doc__)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
np.set_printoptions(linewidth=400)    # screen size
np.set_printoptions(threshold=np.inf) # print all numpy output
dfTrain = pd.read_csv("C://Home//data//file//data01-train.csv",header=None)
dfTest = pd.read_csv("C://Home//data//file//data01-test.csv",header=None)

print(dfTrain[0:10])
arrTrain = dfTrain.as_matrix().astype(int)
print(arrTrain[0:10])
print(arrTrain.shape)

# print(name_dtype)
arrTrain = np.array(arrTrain)
print(arrTrain.dtype)



print(np.unique(arrTrain[:,1])) # binary label 
numClass = np.unique(arrTrain[:,1]).size
print('numClass',numClass)
arrTrainLabels = arrTrain[:,1].astype(int) 
print(arrTrainLabels[0:100])
print('[arrTrainLabels[0,1] : ', np.bincount(arrTrainLabels)) # get frequency of [0,1]
arrTrainFeatures = arrTrain[:,[2,3]]


# svc = LinearSVC(random_state=0)
# svc.fit(featuresTrain, labelTrain)
# print(svc.coef_)
# print(svc.intercept_)

model = SVC(
    C=10.0  # [1,10,100]
    # , cache_size=200
    # , class_weight=None
    # , coef0=0.0
    # , decision_function_shape='ovr'
    # , degree=3
    # , gamma='auto'
    , gamma=10000    # [10,100,1000,10000]
    , kernel='rbf'   # ['linear', 'rbf']
    # , max_iter=-1
    # , probability=False
    # , random_state=None
    # , shrinking=True
    # , tol=0.001
    # , tol=0.001
    # , verbose=False
).fit(arrTrainFeatures, arrTrainLabels)
print(model)
print(model.get_params())

arrTest = dfTest.as_matrix().astype(int)
arrTestLabels = arrTest[0:1000,1]
arrTestFeatures = arrTest[0:1000,[2,3]]
predTestLabels = model.predict(arrTestFeatures)
# print(predTestLabels)


# for i in range(len(arrTestLabels)):
#     print(arrTestLabels[i:i+1,][0],"-", predTestLabels[i,])

pred_score = model.score(arrTestFeatures, arrTestLabels)
# print(pred_score)

arrTestLabel0Count = len(arrTest[np.where(arrTest[:,1]==0)])
# print('Prob(Label=0) :',arrTestLabel0Count/10000)
# print('Prob(Label=1) :',1- arrTestLabel0Count/10000)
# actualTestLabels=arrTest[:,1]
# arrTestActualPred = np.stack([arrTestLabels, pred]).T
accuracyRate = accuracy_score(arrTestLabels, predTestLabels)
print('정확도 = ',accuracyRate)
print('오분류율 = ', 1 - accuracyRate)

