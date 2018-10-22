#!/usr/bin/env /c/Apps/Anaconda3/python

"""
Naive Bayes - Multinomial
Program Code Name : naive-bayes-MultinormialNB.py
Author : 이이백(yibeck.lee@gmail.com)
"""
print(__doc__)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn import tree

print(__doc__)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import tree

np.set_printoptions(linewidth=400)    # screen size
np.set_printoptions(threshold=np.inf) # print all numpy output


dfTrain = pd.read_csv("C://Home//data//file//data01-train.csv",header=None)
nObsDfTrain = len(dfTrain)
print(nObsDfTrain)
dfTest = pd.read_csv("C://Home//data//file//data01-test.csv",header=None)
nObsDfTest = len(dfTest)
print(nObsDfTest)
print(dfTrain[0:10]) # 10개의 관찰치 출력
print(dfTrain.columns.values)
print(dfTrain[0:2][[1,2,12]])

# array index :     0    1    2    3   4     5    6    7    8    9  10     11    12
dfTrain.columns = ['Y1','Y2','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11']
dfTest.columns  = ['Y1','Y2','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11']
print(dfTrain.columns.values)
arrTrain = dfTrain.as_matrix().astype(int)


arrTrainLabels = arrTrain[:,1:2].astype(int) 
print(arrTrainLabels)
arrTrainFeatures = arrTrain[:,[2,3,4,5]]
print(arrTrainFeatures)

model = MultinomialNB().fit(arrTrainFeatures, arrTrainLabels)
print(model)
print(model.get_params())

arrTest = dfTest.as_matrix().astype(int)
arrTestLabels = arrTest[0:1000,1]
arrTestFeatures = arrTest[0:1000,[2,3,4,5]]

pred_proba = model.predict_proba(arrTestFeatures)
print(pred_proba)

pred = model.predict(arrTestFeatures)
print(pred)

for i in range(len(arrTestLabels)):
	print(arrTestLabels[i:i+1,][0],"-", pred[i,])

pred_score = model.score(arrTestFeatures, arrTestLabels)
print(pred_score)

arrTestLabel0Count = len(arrTest[np.where(arrTest[:,1]==0)])
print('Prob(Label=0) :',arrTestLabel0Count/10000)
print('Prob(Label=1) :',1- arrTestLabel0Count/10000)
# actualTestLabels=arrTest[:,1]
# arrTestActualPred = np.stack([arrTestLabels, pred]).T
accuracyRate = accuracy_score(arrTestLabels, pred)
print('정확도 = ',accuracyRate)
print('오분류율 = ', 1 - accuracyRate)
"""
"""

