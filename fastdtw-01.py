#!/usr/bin/env /c/Apps/Anaconda3/python

import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


x = np.array([
    [1,1]
   ,[2,2]
   ,[3,3]
   ,[4,4]
   ,[5,5]
   ])
y = np.array([
    [0,0]
   ,[2,2]
   ,[3,3]
   ,[4,4]
   ,[5,5]
   ])
distance, path = fastdtw(x,y,dist=euclidean)
print(distance)



