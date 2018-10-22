#!/usr/bin/env /c/Apps/Anaconda3/python

"""
SVM - Calculator
Program Code Name : svm-calculator-01.py
Author : 이이백(yibeck.lee@gmail.com)
"""

import numpy as np
from sklearn.svm import SVC, SVR
from numpy import random
num_list = range(0,100)
_int = random.choice(num_list,1)
# print(_int[0])
x = np.array([0,0])
for i in range(5000):
	_int1 = random.choice(num_list,1)[0]
	_int2 = random.choice(num_list,1)[0]
	x = np.vstack((x,[_int1,_int2]))
	# print(i)
y = x[:,0] - x[:,1]

svc = SVC(C=1.0, gamma=0.9)
svc.fit(x,y)
print(svc)
print(svc.predict([[30,10]]))

# svr = SVR(C=10.0,gamma=0.7).fit(x,y)
# print(svr.predict([50,11]))
