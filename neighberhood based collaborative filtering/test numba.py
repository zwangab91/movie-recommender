# playing around 

import numpy as np 
import numba
import os

# for debugging
import timeit
from sys import getsizeof

num_movies = 17770

prefix = '../data/um'


from functools import partial

# location of the current file
here = os.path.dirname(os.path.abspath(__file__))

# load index and all data
arrPar = os.path.join(here, prefix + '/parTraining.npy')
arrPar = np.load(arrPar)


def calc_pearson(userIds, movieIds, ratings, num_movies):
    corrs = np.zeros((num_movies, num_movies))

    for uId, mId, r in zip(userIds, movieIds, ratings):
        corrs[mId-1, mId-1] += r / uId 

    return corrs

@numba.jit('f4[:,:](uint32[:],uint16[:], int8[:], u8)')
def calc_pearson_jit(userIds, movieIds, ratings, num_movies):
    corrs = np.zeros((num_movies, num_movies))

    for uId, mId, r in zip(userIds, movieIds, ratings):
        corrs[mId-1, mId-1] += r / uId 

    return corrs

corr1 = calc_pearson(arrPar['userid'], arrPar['movieid'], arrPar['rating'], num_movies)
corr2 = calc_pearson_jit(arrPar['userid'], arrPar['movieid'], arrPar['rating'], num_movies)

diff_corr = corr1 - corr2
print(np.amax(diff_corr))
print(np.amax(corr1)) 

numTimesP = 3
timetaken = timeit.Timer(partial(calc_pearson, arrPar['userid'], arrPar['movieid'], arrPar['rating'], num_movies)).timeit(number=numTimesP)
print(timetaken)

numTimesN = 100
timetaken = timeit.Timer(partial(calc_pearson_jit, arrPar['userid'], arrPar['movieid'], arrPar['rating'], num_movies)).timeit(number=numTimesN)
print(timetaken)

