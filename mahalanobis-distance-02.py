#!/usr/bin/env /c/Apps/Anaconda3/python
"""
Mahalanobis - get Similarity Distance
Program Code Name : mahalanobis.py
Author : 이이백(yibeck.lee@gmail.com)
"""
print(__doc__)

import scipy.spatial.distance.mahalanobis

x = [1,2,3,4,5,5,100]
y = [1,2,3,4,1,1,10]
# Remove Outlier
dist1 = mahalanobis.MD_removeOutliers(x, y)
print('removed :', dist1)