#!/usr/bin/env /c/Apps/Anaconda3/python

"""
knn Classification
Author : 이이백(yibeck.lee@gmail.com)
"""

print(__doc__)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

np.set_printoptions(linewidth=400)    # screen size
np.set_printoptions(threshold=np.inf) # print all numpy output
dfTrain = pd.read_csv("C://Home//data//file//data01-train.csv",header=None)
dfTest = pd.read_csv("C://Home//data//file//data01-test.csv",header=None)
# print(dfTrain[0:10])
# print(dfTrain.columns.values)
# print(dfTrain[0:2][[1,2,12]])
dfTrain.columns = ['Y1','Y2','X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']
# print(dfTrain.columns.values)
dfTest.columns = ['Y1','Y2','X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']
# print(dfTest.columns[[2,3]])
n_neighbors=14
model = neighbors.KNeighborsClassifier(
     n_neighbors=n_neighbors
    ,weights='distance'
    ,p=2
    ).fit(dfTrain[['X3','X4','X5']], dfTrain['Y2'])
    # ).fit(dfTrain[['X1','X2']], dfTrain['Y2'])

print(model)

pred = model.predict(dfTest[['X3','X4','X5']])
# pred = model.predict(dfTest[['X1','X2']])
# print(pred.dtype)
# for i in range(pred.size):
#   print('{0:4d}: {1:1.0f} - {2:1.0f}'.format(i,dfTest['Y2'][i],pred[i]))

# print(dfTest['Y2'])

print(dfTest['Y2'].count())
dfFreq = pd.value_counts(dfTest['Y2']).to_frame().reset_index()
print(dfFreq)

# print(dfTest['Y2'].describe())

cmatrix = confusion_matrix(y_true=dfTest['Y2'],y_pred=pred)

print(cmatrix)

true_negative, false_positive, false_negative, true_positive = confusion_matrix(y_true=dfTest['Y2'],y_pred=pred).ravel()
print('true_negative, false_positive, false_negative, true_positive\n', true_negative, false_positive, false_negative, true_positive)

print(classification_report(y_true=dfTest['Y2'], y_pred=pred))

accuracyRate = accuracy_score(dfTest['Y2'], pred,)
print('정확도 = ',accuracyRate)
print('오분류율 = ', 1 - accuracyRate)



"""
print(__doc__)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors
from sklearn.metrics import accuracy_score


n_neighbors = 15
h = .02  # step size in the mesh
# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


np.set_printoptions(linewidth=400)    # screen size
np.set_printoptions(threshold=np.inf) # print all numpy output
dfTrain = pd.read_csv("C://Home//GitHub//deepbig-ng2-login//src//assets//resources//data//data01-train.csv",header=None)
dfTest = pd.read_csv("C://Home//GitHub//deepbig-ng2-login//src//assets//resources//data//data01-test.csv",header=None)
arrTrain = dfTrain.as_matrix().astype(int)
labelTrain = arrTrain[:,1].astype(int) 
featuresTrain = arrTrain[:,2:4]

for weights in ['uniform', 'distance']:
    # we create an instance of Neighbours Classifier and fit the data.
    knnc = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    knnc.fit(featuresTrain, labelTrain)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = featuresTrain[:, 0].min() - 1, featuresTrain[:, 0].max() + 1
    y_min, y_max = featuresTrain[:, 1].min() - 1, featuresTrain[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = knnc.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    print('Z : ', Z)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(featuresTrain[:, 0], featuresTrain[:, 1], c=labelTrain, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))

plt.show()
"""