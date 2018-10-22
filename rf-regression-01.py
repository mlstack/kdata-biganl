#!/usr/bin/env /c/Apps/Anaconda3/python

"""
Random Forest - Regression
Program Code Name : rf-regression-01.py
Author : 이이백(yibeck.lee@gmail.com)
"""

print(__doc__)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
np.set_printoptions(linewidth=400)    # screen size
np.set_printoptions(threshold=np.inf) # print all numpy output
dfTrain = pd.read_csv("C://Home//data//file//data01-train.csv",header=None)
dfTest = pd.read_csv("C://Home//data//file//data01-test.csv",header=None)
print(dfTrain[0:10])
arrTrain = dfTrain.as_matrix().astype(int)
print(arrTrain[0:10])
print(arrTrain.shape)

arrTrainLabels = arrTrain[:,0].astype(int) 
print(arrTrainLabels[0:100])
arrTrainFeatures = arrTrain[:,2:]


# Train and make model
# rfr = DecisionTreeRegressor().fit(featuresTrain, labelTrain)
model = RandomForestRegressor().fit(arrTrainFeatures, arrTrainLabels)
print(model)

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

arrTest = dfTest.as_matrix().astype(int)

arrTestFeatures = arrTest[:,2:]
predTestLabels = model.predict(arrTestFeatures)
print(predTestLabels)

arrTestLabels=arrTest[:,0]

arrLabelTest = np.stack([arrTestLabels, predTestLabels]).T
print(arrLabelTest)
print(arrLabelTest[0:10,0])
print(arrLabelTest[0:10,1])

plt.scatter(arrLabelTest[0:10000,0], arrLabelTest[0:10000,1])
plt.show()

