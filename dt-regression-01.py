#!/usr/bin/env /c/Apps/Anaconda3/python

"""
Decision Tree Regression
Program Code Name : dt-regression-01.py
Author : 이이백(yibeck.lee@gmail.com)
"""
print(__doc__)
import pandas as pd
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

import graphviz 
from sklearn import tree

np.set_printoptions(linewidth=400)    # screen size
np.set_printoptions(threshold=np.inf) # print all numpy output

dfTrain = pd.read_csv("C://Home//data//file//data01-train.csv"
    ,header=None)
dfTest = pd.read_csv("C://Home//data//file//data01-test.csv"
    ,header=None)
nObsDfTrain = len(dfTrain)
nObsDfTest = len(dfTest)
print(nObsDfTrain,nObsDfTest)

print(dfTrain[0:10]) # 10개의 관찰치 출력
print(dfTrain.columns.values)
print(dfTrain[0:2][[1,2,12]])
# array index :     0    1    2    3   4     5    6    7    8    9  10     11    12
dfTrain.columns = ['Y1','Y2','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11']
dfTest.columns  = ['Y1','Y2','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11']
print(dfTrain.columns.values)
print(dfTrain[['Y1','Y2']][:10])


# dfTrainFeatures = dfTrain[['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11']]
dfTrainFeatures = dfTrain[['V10','V1','V2','V3']]
dfTrainLabels = dfTrain[['Y1']]

obsFrom = 0
obsTo = nObsDfTest
# dfTestFeatures = dfTest[['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11']]
dfTestFeatures = dfTest[['V10','V1','V2','V3']]
dfTestLabels = dfTest[['Y1']]

model = DecisionTreeRegressor(
    criterion='mse'
,	max_depth=2
    ).fit(dfTrainFeatures, dfTrainLabels)
print(model)

# numpy.array로 변환
arrTestFeatures = dfTestFeatures.values.astype(int)
testLabels = dfTestLabels.values.astype(int)
# print(type(arrTestFeatures[:,0]))
# print(arrTestFeatures[0:1,0:2])
testLabelsPrediction = model.predict(arrTestFeatures)
print(testLabelsPrediction[0:10	])
testLabelsPrediction=testLabelsPrediction.astype(int)
for i in range(10):
    print(testLabels[i],[testLabelsPrediction[i]])


with open("model_regression.txt", "w") as f:
	f = tree.export_graphviz(
		model
	, 	out_file=f
	, 	feature_names=['V10','V1','V2','V3']
	)
	graph = graphviz.Source(f) 
	# graph.render('model') 

dot_data = tree.export_graphviz(model, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("dtr") 




importances = model.feature_importances_
print('feature importances : ', importances)
indices = np.argsort(importances)[::-1]
print('indices : ', indices)
sorted_importances = np.argsort(importances)
print(sorted_importances)
# columnNames = dfTrain[['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11']].columns.values
columnNames = dfTrain[['V10','V1','V2','V3']].columns.values
print(columnNames)
print("변수 중요도")
print(sorted(zip(map(lambda x: round(x, 4), model.feature_importances_), columnNames), reverse=True))
padding = np.arange(len(columnNames)) + 0.5

plt.figure(1)
plt.barh(padding, importances[sorted_importances])
plt.yticks(padding, columnNames[sorted_importances])
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(testLabels,testLabelsPrediction)
print('mse = {:12,.0f}'.format(mse))


"""
"""
