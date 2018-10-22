#!/usr/bin/env /c/Apps/Anaconda3/python

"""
GLM - Lasso
Program Code Name : glm-lass-01.py
Author : 이이백(yibeck.lee@gmail.com)
"""
print(__doc__)
import pandas as pd
from pandas import Series
import numpy as np
from sklearn import linear_model as lm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

np.set_printoptions(linewidth=400)    # screen size
np.set_printoptions(threshold=np.inf) # print all numpy output
dfTrain = pd.read_csv("C://Home//data//file//data01-train.csv",header=None)
dfTest = pd.read_csv("C://Home//data//file//data01-test.csv",header=None)
print(dfTrain.columns.values)
print(dfTrain[0:2][[1,2,12]])
dfTrain.columns = ['Y1','Y2','X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']
dfTest.columns = ['Y1','Y2','X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']

print('dfTrain.columns : ',  dfTrain.columns)
print('dfTrain.columns.values : ',  dfTrain.columns.values)

print('[alpha=0]')
glm_lasso = lm.Lasso(
	alpha=0
	).fit(dfTrain[['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']],dfTrain['Y1'])
predLabelTrain = glm_lasso.predict(dfTrain[['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']])
r2 = r2_score(dfTrain['Y1'],predLabelTrain)
print('r-squared = ', r2)
mse = mean_squared_error(dfTrain['Y1'],predLabelTrain)
print('mse = {:12,.0f}'.format(mse))
lasso_coef =  Series(
	 glm_lasso.coef_
	,dfTrain[['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']].columns
	).sort_values()
print('type of lasso_coef:', type(lasso_coef))
# print(lasso_coef.size)
print('beta coefficient of variables')
for i in range(lasso_coef.size):
	print('{:4s} : {:10.8f}'.format(lasso_coef.index.values[i],lasso_coef.values[i]))
plt.figure(1)
lasso_coef.plot(kind='bar',grid=True)
plt.savefig("lasso_alpha0.png")
print('[alpha=100]')
glm_lasso = lm.Lasso(
	alpha=10
	).fit(dfTrain[['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']],dfTrain['Y1'])
predLabelTrain = glm_lasso.predict(dfTrain[['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']])
r2 = r2_score(dfTrain['Y1'],predLabelTrain)
print('r-squared = ', r2)
mse = mean_squared_error(dfTrain['Y1'],predLabelTrain)
print('mse = {:12,.0f}'.format(mse))
lasso_coef =  Series(
	 glm_lasso.coef_
	,dfTrain[['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']].columns
	).sort_values()
print('type of lasso_coef:', type(lasso_coef))
# print(lasso_coef.size)
print('beta coefficient of variables')
for i in range(lasso_coef.size):
	print('{:4s} : {:10.8f}'.format(lasso_coef.index.values[i],lasso_coef.values[i]))
plt.figure(2)
lasso_coef.plot(kind='bar',grid=True)
plt.savefig("lasso_alpha100.png")

plt.show()



"""
print(__doc__)
import pandas as pd
import numpy as np
from sklearn import linear_model as lm
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=400)    # screen size
np.set_printoptions(threshold=np.inf) # print all numpy output
dfTrain = pd.read_csv("C://Home//GitHub//deepbig-ng2-login//src//assets//resources//data//data01-train.csv",header=None)
dfTest = pd.read_csv("C://Home//GitHub//deepbig-ng2-login//src//assets//resources//data//data01-test.csv",header=None)

arrTrain = dfTrain.as_matrix().astype(int)

labelTrain = arrTrain[:,0]

featuresTrain = arrTrain[:,[11,12,10,9,8,7,6,5,4,3,2]]
glm_lasso = lm.Lasso(alpha=.1).fit(featuresTrain,labelTrain)

arrTest = dfTest.as_matrix().astype(int)
predTest = arrTest[:,0:]
featuresTest = arrTest[:,[11,12,10,9,8,7,6,5,4,3,2]]

predLabelTest = glm_lasso.predict(featuresTest)

# print(predLabelTest)

actualLabelTest=arrTest[:,0]
arrLabelTest = np.stack([actualLabelTest, predLabelTest]).T
nrow = arrLabelTest.shape[0]
for i in range(nrow):
	print('actual={:10,.0f}  pred={:10,.0f}'.format(arrLabelTest[i,0],arrLabelTest[i,1]))

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(actualLabelTest,predLabelTest)
print('{:12,.0f}'.format(mse))

plt.figure(1)
plt.scatter(arrTest[:,6],actualLabelTest)
plt.show() 

plt.figure(2)
plt.scatter(actualLabelTest,actualLabelTest)
plt.scatter(actualLabelTest,predLabelTest)
plt.show()
"""