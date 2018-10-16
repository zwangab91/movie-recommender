# playing around 

import numpy as np 
import os

# for debugging
import timeit
from sys import getsizeof

prefix = '../data/um'

# location of the current file
here = os.path.dirname(os.path.abspath(__file__))

# load index and all data
arrInd = os.path.join(here, prefix + '/allIndices.npy')
arrInd = np.load(arrInd)
arrAll = os.path.join(here, prefix + '/allData.npy')
arrAll = np.load(arrAll)

# get only elements that are in the training set
filteredArr = np.isin(arrInd, [1])
arrTraining = arrAll[filteredArr]

savename = os.path.join(here, prefix + '/parTraining')
np.save(savename, arrTraining[:100000])
 