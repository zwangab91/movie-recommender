import numpy as np 
import os
from scipy import stats
import math

# for debugging
import timeit
from sys import getsizeof
from functools import partial

num_movies, num_users = 17770, 458293
# location of the current file
here = os.path.dirname(os.path.abspath(__file__))

# load um data
prefix = '../data/um'
trainDataUM = os.path.join(here, prefix + '/trainingData.npy')
trainDataUM = np.load(trainDataUM)

num_data = len(trainDataUM)
# load mu data
prefix = '../data/mu'
trainDataMU = os.path.join(here, prefix + '/trainingData.npy')
trainDataMU = np.load(trainDataMU)


def get_overlap_1to1(mov1, mov2, ends_movs, userIds, ratings):
    # find the users who rated movie mov1
    if mov1 == 1:
        mov1_ind_start = 0
    else:
        mov1_ind_start = ends_movs[mov1-2]
    mov1_ind_end = ends_movs[mov1-1]
    users_rated_mov1 = userIds[mov1_ind_start:mov1_ind_end]

    # find the users who rated movie mov2
    if mov2 == 1:
        mov2_ind_start = 0
    else:
        mov2_ind_start = ends_movs[mov2-2]
    mov2_ind_end = ends_movs[mov2-1]
    users_rated_mov2 = userIds[mov2_ind_start:mov2_ind_end]

    # find the intersection between them
    rangeMov1 = np.arange(mov1_ind_start, mov1_ind_end)
    rangeMov2 = np.arange(mov2_ind_start, mov2_ind_end)

    mask12 = np.in1d(users_rated_mov1, users_rated_mov2, assume_unique=True)
    mask21 = np.in1d(users_rated_mov2, users_rated_mov1, assume_unique=True)

    # the arrays that will be passed to pearson correlation function
    x = ratings[rangeMov1[mask12]]
    y = ratings[rangeMov2[mask21]]

    # x = x.astype(float)
    # y = y.astype(float)
    # if len(x) != len(y):
    #     raise ValueError("x and y should have the same length!")

    if len(x) > 1:
        corr = stats.pearsonr(x, y)[0]
        if math.isnan(corr):
            return -10
        else:
            return corr
    else:
        return -10

# @numba.jit('void(u8, int32[:], uint32[:], int8[:], uint16[:])')      
def get_overlap_slow(mov, ends_movs, userIds, ratings, range_movs):
    pearsons = np.zeros(range_movs[-1]) 

    for ind, mov2 in enumerate(range_movs):
        pearsons[ind] = get_overlap_1to1(mov, mov2, ends_movs, userIds, ratings)

    return pearsons
