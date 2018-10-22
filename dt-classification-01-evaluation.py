#!/usr/bin/env /c/Apps/Anaconda3/python

"""
Decision Tree Classification
Program code name : dt-classification-01.py 
Author : 이이백(yibeck.lee@gmail.com)
"""

# print(__doc__)

import pandas as pd
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score \
, confusion_matrix,classification_report
from sklearn import tree # tree graph

np.set_printoptions(linewidth=400)    # screen size
np.set_printoptions(threshold=np.inf) # print all numpy output

dfTrain = pd.read_csv("C://Home//data//file//data01-train.csv"
    ,header=None)
dfTest = pd.read_csv("C://Home//data//file//data01-test.csv"
    ,header=None)

print(dfTrain[0:10]) # 10개의 관찰치 출력
print(dfTrain.columns.values)
print(dfTrain[0:2][[1,2,12]])
# array index :     0    1    2    3   4     5    6    7    8    9  10     11    12
dfTrain.columns = ['Y1','Y2','V1','V2','V3','V4','V5','V6'
,'V7','V8','V9','V10','V11']
dfTest.columns  = ['Y1','Y2','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11']
print(dfTrain.columns.values)

dfTrainFeatures = dfTrain[['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11']]
dfTrainLabels = dfTrain['Y2']

model = DecisionTreeClassifier(
    max_depth=2
    ).fit(dfTrainFeatures
         ,dfTrainLabels)

print(model)

# conda install graphviz
# pip install graphviz
# download and install : graphviz-2.38.msi
# [download link] https://graphviz.gitlab.io/_pages/Download/Download_windows.html
# [set path] C:\Apps\Graphviz2.38\bin
import graphviz 

dot_data = tree.export_graphviz(
    model
,   out_file=None
,   feature_names=['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11']
,   class_names=['good','bad']
)
graph = graphviz.Source(dot_data)
graph.render("decition-tree") 

dfTestFeatures = dfTest[['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11']]
dfTestLabels = dfTest['Y2']

dfTestLabelsPred = model.predict(dfTestFeatures)
# print(dfTestLabelPred)


print("[Model Evaluation]\n")

print("[Number of Test Observations] {:,}".format(dfTestLabels.count()))
dfStatsFreq = pd.value_counts(dfTest['Y2']).to_frame().reset_index()
print("[Frequence Table]\n",dfStatsFreq)

confusionMatrix = confusion_matrix(
    y_true=dfTestLabels
,   y_pred=dfTestLabelsPred)
print(confusionMatrix)

true_negative, false_positive, false_negative, true_positive \
= confusion_matrix(
    y_true=dfTestLabels
,   y_pred=dfTestLabelsPred
    ).ravel()

accuracyRate = accuracy_score(
    y_true=dfTestLabels
,   y_pred=dfTestLabelsPred
)
print('정확도 = ',accuracyRate)
print('오분류율 = ', 1 - accuracyRate)


importances = model.feature_importances_
sorted_importances = np.argsort(importances)
print(sorted_importances)

columnNames = dfTrain[['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11']].columns.values
print(columnNames)
print("변수 중요도")
print(sorted(zip(map(lambda x: round(x, 4), model.feature_importances_), columnNames), reverse=True))
padding = np.arange(len(columnNames)) + 0.5

plt.figure(1)
plt.barh(padding, importances[sorted_importances], align='center')
# plt.barh(padding, importances[sorted_importances])
plt.yticks(padding, columnNames[sorted_importances])
plt.title("Feature Importance")
plt.xlabel("Relative Importance")
plt.ylabel("Features")
# plt.show()

print("[ROC] \n")
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(
    y_true=dfTestLabels
,   y_score=dfTestLabelsPred
)
print("fpr",fpr)
print("tpr",tpr)
plt.figure(2)
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1])
plt.title("ROC")
plt.xlabel("fpr")
plt.ylabel("tpr")
plt.show()

print('true_negative, false_positive, false_negative, true_positive\n'
    , true_negative, false_positive, false_negative, true_positive)
print(classification_report(
    y_true=dfTestLabels
,   y_pred=dfTestLabelsPred
))

"""
"""

