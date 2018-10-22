#!/usr/bin/env /c/Apps/Anaconda3/python
"""
Mahalannobis Distance
"""
import numpy as np
from scipy.spatial import distance
a = np.array([[0, 0, 0],
               [0, 0, 1],
               [0, 1, 0],
               [0, 1, 1],
               [1, 0, 0],
               [1, 0, 1],
               [1, 1, 0],
               [1, 1, 1]])
b = np.array([[ 0.1,  0.2,  0.4]])
mahalanobis_distance = distance.cdist(a, b, 'mahalanobis')
print(mahalanobis_distance)