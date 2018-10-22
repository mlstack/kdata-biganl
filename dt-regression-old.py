#!/usr/bin/env /c/Apps/Anaconda3/python

"""
Decision Tree Regression
Program Code Name : dt-regression-01.py
Author : 이이백(yibeck.lee@gmail.com)
"""
print(__doc__)
import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
np.set_printoptions(linewidth=400)    # screen size
np.set_printoptions(threshold=np.inf) # print all numpy output
np.set_printoptions(linewidth=400)    # screen size
np.set_printoptions(threshold=np.inf) # print all numpy output

dfTrain = pd.read_csv("C://Home//data//file//data01-train.csv"
    ,header=None)
nObsDfTrain = len(dfTrain)
print(nObsDfTrain)
dfTest = pd.read_csv("C://Home//data//file//data01-test.csv"
    ,header=None)
nObsDfTest = len(dfTest)
print(nObsDfTest)

print(dfTrain[0:10]) # 10개의 관찰치 출력
print(dfTrain.columns.values)
print(dfTrain[0:2][[1,2,12]])
# array index :     0    1    2    3   4     5    6    7    8    9  10     11    12
dfTrain.columns = ['Y1','Y2','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11']
dfTest.columns  = ['Y1','Y2','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11']
print(dfTrain.columns.values)


dfTrainFeatures = dfTest[['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11']]
dfTrainLabels = dfTest[['Y1']]

obsFrom = 0
obsTo = nObsDfTest
dfTestFeatures = dfTest[['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11']][obsFrom:obsTo]
dfTestLabels = dfTest[['Y1']][obsFrom:obsTo]
# number of observation
# print(len(obsTestFeatures))

# dataframe to numpy array
# arrTrain = dfTrain.as_matrix().astype(int)
# print(arrTrain[0:1,])

# change variable name in numpy array
# name_dtype = [
#       ('continuousLabel', int)
#     , ('binaryLabel', int)
#     , ('X0', int)
#     , ('X1', int)
#     , ('X2', int)
#     , ('X3', int)
#     , ('X4', int)
#     , ('X5', int)
#     , ('X6', int)
#     , ('X7', int)
#     , ('X8', int)
#     , ('X9', int)
#     , ('X10', int)
#     ]
# print(name_dtype)

# arrTrain = np.array(arrTrain, dtype = name_dtype)

# print(arrTrain.dtype)
# print(arrTrain.shape)
# print(arrTrain)

# arrTrainFeatures = arrTrain[:,2:]
# print(arrTrainFeatures[0:1,])

# arrTrainLabels = arrTrain[:,0]
# print(arrTrainLabels[0:1])

# Train and make model
# model = DecisionTreeRegressor().fit(featuresTrain, labelTrain)
model = DecisionTreeRegressor(
      criterion='mse'
    , max_depth=None
    , max_features=None
    , max_leaf_nodes=None
    , min_impurity_split=0.01
    , min_samples_leaf=10
    , min_samples_split=20
    , min_weight_fraction_leaf=0.0
    , presort=False
    , random_state=None
    , splitter='best'
    ).fit(dfTrainFeatures, dfTrainLabels)
print(model)

# importances = model.feature_importances_
# print('feature importances : ', importances)
# indices = np.argsort(importances)[::-1]
# print('indices : ', indices)

# for f in range(dfTrainFeatures.shape[1]):
#     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# f, ax = plt.subplots(figsize=(11, 9))
# plt.title("Feature ranking", fontsize = 20)
# plt.bar(range(featuresTrain.shape[1]), importances[indices],
#     color="b", 
#     align="center")
# plt.xticks(range(featuresTrain.shape[1]), indices)
# plt.xlim([-1, featuresTrain.shape[1]])
# plt.ylabel("importance", fontsize = 18)
# plt.xlabel("index of the feature", fontsize = 18)
# plt.show()



names = dfTrain[['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11']].columns.values
print(names)
importances = model.feature_importances_
sorted_importances = np.argsort(importances)
print(sorted_importances)

print("변수 중요도")
print(sorted(zip(map(lambda x: round(x, 4), model.feature_importances_), names), reverse=True))
padding = np.arange(len(names)) + 0.5
plt.barh(padding, importances[sorted_importances], align='center')
# plt.barh(padding, importances[sorted_importances])
plt.yticks(padding, names[sorted_importances])
plt.xlabel("Relative Importance")
plt.title("Feature Importance")
plt.show()



arrTest = dfTest.as_matrix().astype(int)

arrTestFeatures = arrTest[:,2:]
predTestLabels = model.predict(arrTestFeatures)
#print(predLabelTest)
actualTestLabels=arrTest[:,0]
arrTestLabels = np.stack([actualTestLabels, predTestLabels]).T
print(arrTestLabels[0:10,0:1])
print(arrTestLabels[0:10,1:2])

plt.scatter(arrTestLabels[0:10000,0:1], arrTestLabels[0:10000,1:2])
plt.show()


import graphviz 
dot_data = tree.export_graphviz(model, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("dtr") 
# dot_data = tree.export_graphviz(model, out_file=None, 
#                          feature_names=['X0','X1','X2','X3','X4','X5','X6','X7','X8','X0','X10'],  
#                          # class_names=iris.target_names,  
#                          filled=True, rounded=True,  
#                          special_characters=True)  
# graph = graphviz.Source(dot_data)  
# graph
