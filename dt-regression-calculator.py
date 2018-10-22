#!/usr/bin/env /c/Apps/Anaconda3/python

"""
Understing Decisiont Tree  - Calculator
Author : 이이백(yibeck.lee@gmail.com)
"""
print(__doc__)

from sklearn import tree
X = [
	 [1, 1, 2]
	,[1, 2, 2]
	,[2, 1, 2]
	,[2, 2, 2]
	,[3, 3, 2]
	,[3, 4, 2]
	]

Y = [1,0,5,4,7,6]
dtc = tree.DecisionTreeClassifier()
dtc = dtc.fit(X, Y)
out1 = dtc.predict([[2, 3, 2]])
print("DTC - [2,3]", out1)



out1 = dtc.predict([[3, 2]])
print("DTC - [3,2]", out1)


X = [
	 [0., 0.]
	,[0.1, 0.1]
	,[0., 1.]
	,[1., 0.]
	,[1., 1.]
	,[1, 1.3]
	,[1, 1.5]
	,[2., 2.]
	]
# Y = [0, 1]
Y = [0.,0.2, 1.,1.,2.,2.3,2.5, 4.]

dtr = tree.DecisionTreeRegressor()
dtr = dtr.fit(X, Y)

out2 = dtr.predict([[1., 1.4]])
print('DTR - [1.,1.4] = ', out2)

out3 = dtr.predict([[2., 5.9]])
print('DTR - [2.,5.9] = ', out3)

"""

"""

"""





















"""