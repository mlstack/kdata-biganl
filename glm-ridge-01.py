#!/usr/bin/env /c/Apps/Anaconda3/python

"""
GLM - ridge
Program Code Name : glm-ridge-01.py
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
# dfTest = pd.read_csv("C://Home//GitHub//deepbig-ng2-login//resources//data//data01-train.csv",header=None)
print(dfTrain.columns.values)
print(dfTrain[0:2][[1,2,12]])
dfTrain.columns = ['Y1','Y2','X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']
dfTest.columns = ['Y1','Y2','X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']

print('dfTrain.columns : ',  dfTrain.columns)
print('dfTrain.columns.values : ',  dfTrain.columns.values)

print('[alpha=0]')
glm_ridge = lm.Ridge(
	alpha=0
	).fit(dfTrain[['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']],dfTrain['Y1'])
predLabelTrain = glm_ridge.predict(dfTrain[['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']])
r2 = r2_score(dfTrain['Y1'],predLabelTrain)
print('r-squared = ', r2)
mse = mean_squared_error(dfTrain['Y1'],predLabelTrain)
print('mse = {:12,.0f}'.format(mse))
ridge_coef =  Series(
	 glm_ridge.coef_
	,dfTrain[['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']].columns
	).sort_values()
print('type of ridge_coef:', type(ridge_coef))
# print(ridge_coef.size)
print('beta coefficient of variables')
# for idx,value in enumerate(ridge_coef):
# 	print('{} : {:10.8f}'.format(idx,value))
for i in range(ridge_coef.size):
	print('{:4s} : {:10.8f}'.format(ridge_coef.index.values[i],ridge_coef.values[i]))

plt.figure(1)
ridge_coef.plot(kind='bar',grid=True)
plt.savefig("ridge_alpha0.png")




print('[alpha=100]')
glm_ridge = lm.Ridge(
	alpha=100
	).fit(dfTrain[['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']],dfTrain['Y1'])
predLabelTrain = glm_ridge.predict(dfTrain[['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']])
r2 = r2_score(dfTrain['Y1'],predLabelTrain)
print('r-squared = ', r2)
mse = mean_squared_error(dfTrain['Y1'],predLabelTrain)
print('mse = {:12,.0f}'.format(mse))
ridge_coef =  Series(
	 glm_ridge.coef_
	,dfTrain[['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']].columns
	).sort_values()
print('type of ridge_coef:', type(ridge_coef))
# print(ridge_coef.size)
print('beta coefficient of variables')
for i in range(ridge_coef.size):
	print('{:4s} : {:10.8f}'.format(ridge_coef.index.values[i],ridge_coef.values[i]))
plt.figure(2)
ridge_coef.plot(kind='bar',grid=True)
plt.savefig("ridge_alpha100.png")

plt.show()

"""
print('[alpha=1]')
glm_ridge = lm.Ridge(
	alpha=1000
	).fit(dfTrain[['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']],dfTrain['Y1'])
predLabelTrain = glm_ridge.predict(dfTrain[['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']])
r2 = r2_score(dfTrain['Y1'],predLabelTrain)
print('r-squared = ', r2)
mse = mean_squared_error(dfTrain['Y1'],predLabelTrain)
print('mse = {:12,.0f}'.format(mse))
ridge_coef =  Series(glm_ridge.coef_,dfTrain[['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']].columns).sort_values()
# print(ridge_coef.size)
print('beta coefficient of variables')
for i in range(ridge_coef.size):
	print('{:4s} : {:10.8f}'.format(ridge_coef.index.values[i],ridge_coef.values[i]))


print('[Test, alpha=0]')
glm_ridge = lm.Ridge(
	alpha=0
	).fit(dfTrain[['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']],dfTrain['Y1'])
predLabelTest = glm_ridge.predict(dfTest[['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']])
mse = mean_squared_error(dfTest['Y1'],predLabelTest)
print('mse = {:12,.0f}'.format(mse))

print('[Test, alpha=10]')
glm_ridge = lm.Ridge(
	alpha=10
	).fit(dfTrain[['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']],dfTrain['Y1'])
predLabelTest = glm_ridge.predict(dfTest[['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']])
mse = mean_squared_error(dfTest['Y1'],predLabelTest)
print('mse = {:12,.0f}'.format(mse))

# predLabelTest = glm_ridge.predict([1,2,3,40,52,6,7,8,9,100,1100])
# print(predLabelTest)
for i in range(predLabelTest.size):
	print('{0:}: {1:8,.0f} {2:8,.0f}'.format(i,dfTest['Y1'][i],predLabelTest[i]))
"""


# glm_ridge = lm.Ridge(
# 	alpha=0
# 	).fit(dfTrain[['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']],dfTrain['Y1'])
# print(glm_ridge)

# ridge_coef =  Series(glm_ridge.coef_,dfTrain[['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']].columns).sort_values()

# for i in range(11):
# 	print('{:4s} = {:12.10f}'.format(ridge_coef.index.values[i],ridge_coef.values[i]))

# predLabelTest = glm_ridge.predict(dfTest[['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']])
# print(predLabelTest.dtype)
# for i in range(predLabelTest.size):
# 	print(i,predLabelTest[i])





# arrLabelTest = np.stack([dfTrain['Y1'],predLabelTest]).T
# print(arrLabelTest)





"""
arrTrain = dfTrain.as_matrix().astype(int)

labelTrain = arrTrain[:,0]

featuresTrain = arrTrain[:,[11,12,10,9,8,7,6,5,4,3,2]]
glm_ridge = lm.Ridge(alpha=.5).fit(featuresTrain,labelTrain)

arrTest = dfTest.as_matrix().astype(int)
predTest = arrTest[:,0:]
featuresTest = arrTest[:,[11,12,10,9,8,7,6,5,4,3,2]]

predLabelTest = glm_ridge.predict(featuresTest)

# print(predLabelTest)

actualLabelTest=arrTest[:,0]
arrLabelTest = np.stack([actualLabelTest, predLabelTest]).T
nrow = arrLabelTest.shape[0]
for i in range(nrow):
	print('actual={:10,.0f}  pred={:10,.0f}'.format(arrLabelTest[i,0],arrLabelTest[i,1]))

print('train r2',glm_ridge.score(featuresTrain,labelTrain))


print('coef_ :', glm_ridge.coef_)

r2 = r2_score(actualLabelTest,predLabelTest)
print('test  r2 = ', r2)
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
"""
"""