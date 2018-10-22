#!/usr/bin/env /c/Apps/Anaconda3/python

"""
Random Forest - Classification
Program Code Name : rf-classification-01.py
Author : 이이백(yibeck.lee@gmail.com)
"""

print(__doc__)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
np.set_printoptions(linewidth=400)    # screen size
np.set_printoptions(threshold=np.inf) # print all numpy output
dfTrain = pd.read_csv("C://Home//data//file//data01-train.csv",header=None)
dfTest = pd.read_csv("C://Home//data//file//data01-test.csv",header=None)

print(dfTrain[0:10])
arrTrain = dfTrain.as_matrix().astype(int)
# arrTrain = dfTrain.as_matrix()
print(arrTrain[0:10])
print(arrTrain.shape)
arrTrainFeatures = arrTrain[:,2:].astype(int) 

# labelTrain = arrTrain[:,1]
arrTrainLabels = arrTrain[:,1].astype(int) 
print(arrTrainLabels)
print('[arrTrainLabels[0,1] : ', np.bincount(arrTrainLabels)) # get frequency of [0,1]
# Train and make model
model = RandomForestClassifier().fit(arrTrainFeatures, arrTrainLabels)
print(model)

arrTest = dfTest.as_matrix().astype(int)
arrTestLabels = arrTest[:,1:]
arrTestFeatures = arrTest[:,2:]
predTestLabels = model.predict(arrTestFeatures)
print(predTestLabels)
arrTestLabels=arrTest[:,1]
print(arrTestLabels)

arrLabelTest = np.stack([arrTestLabels, predTestLabels]).T
print(arrLabelTest)

accuracyRate = accuracy_score(arrTestLabels, predTestLabels)
print('정확도 = ',accuracyRate)
print('오분류율 = ', 1 - accuracyRate)

importances = model.feature_importances_
print('feature importances : ', importances)
indices = np.argsort(importances)[::-1]

print('indices : ', indices)

for f in range(arrTrainFeatures.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


f, ax = plt.subplots(figsize=(11, 9))
plt.title("Feature ranking", fontsize = 20)
plt.bar(range(arrTrainFeatures.shape[1]), importances[indices],
    color="b", 
    align="center")
plt.xticks(range(arrTrainFeatures.shape[1]), indices)
plt.xlim([-1, arrTrainFeatures.shape[1]])
plt.ylabel("importance", fontsize = 18)
plt.xlabel("index of the feature", fontsize = 18)
plt.show()
