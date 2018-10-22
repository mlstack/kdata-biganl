#!/usr/bin/env /c/Apps/Anaconda3/python

"""
DTW - Dynamic Time Wrap
Program Code Name : dtw-01.py
Author : 이이백(yibeck.lee@gmail.com)
"""
print(__doc__)
from dtaidistance import dtw
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(linewidth=400)    # screen size

s1 = [0, 0, 1, 2, 1, 0, 1, 0, 0]
# s2 = [0, 0, 1, 2, 1, 1, 0, 0, 0]
# s2 = [0, 1, 2, 0, 0, 0, 0, 0, 0]
s2 = [1, 1, 2, 1, 1, 1, 1, 1, 1]
dtw.plot_warping(s1, s2)

plt.plot(s1)
plt.plot(s2)
plt.show()

distance = dtw.distance(s1, s2)
print(distance)
# print(dtw.distance.__doc__)
# distance, matrix = dtw.distances(s1, s2)
# print(distance)
# print(matrix)
# d = dtw.distance_fast(s1, s2)