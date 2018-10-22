#!/usr/bin/env /c/Apps/Anaconda3/python

"""
knn Regression
Program Code Name : knn-regression-01.py
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
n_neighbors=14
model = neighbors.KNeighborsRegressor(
     n_neighbors=n_neighbors
    ,weights='distance'
    ,p=2
    ).fit(dfTrain[['X3','X4','X5']], dfTrain['Y1'])
    # ).fit(dfTrain[['X1','X2']], dfTrain['Y2'])

print(model)
# pred_prob = knnc.predict_proba(dfTrain[['X1','X2']])
# print(pred_prob)
predTestLabel = model.predict(dfTest[['X3','X4','X5']])
# pred = knnc.predict(dfTest[['X1','X2']])
# print(pred.dtype)
for i in range(predTestLabel.size):
  print('{0:4d}: {1:10,.0f} - {2:10,.0f}'.format(i,dfTest['Y1'][i],predTestLabel[i]))

mse = mean_squared_error(dfTrain['Y1'],predTestLabel)
print('mse = {:12,.0f}'.format(mse))



"""
print(__doc__)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors
from sklearn.metrics import accuracy_score


n_neighbors = 5

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


np.set_printoptions(linewidth=400)    # screen size
np.set_printoptions(threshold=np.inf) # print all numpy output
dfTrain = pd.read_csv("C://Home//GitHub//deepbig-ng2-login//src//assets//resources//data//data01-train.csv",header=None)
dfTest = pd.read_csv("C://Home//GitHub//deepbig-ng2-login//src//assets//resources//data//data01-test.csv",header=None)
arrTrain = dfTrain.as_matrix().astype(int)
labelTrain = arrTrain[:,0].astype(int) 
featuresTrain = arrTrain[:,2:3]


for i, weights in enumerate(['uniform', 'distance']):
    knnr = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
    y_ = knnr.fit(featuresTrain, labelTrain).predict(featuresTrain)

    plt.subplot(2, 1, i + 1)
    plt.scatter(featuresTrain, labelTrain, c='k', label='data')
    plt.plot(featuresTrain, y_, c='g', label='prediction')
    plt.axis('tight')
    plt.legend()
    plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors,
                                                                weights))

plt.show()
"""