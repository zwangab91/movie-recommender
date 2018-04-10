# playing around 

import numpy as np 
import numba
import os

# for debugging
import timeit
from sys import getsizeof

num_movies, num_users = 17770, 458293

prefix = '../data/um'


from functools import partial

# location of the current file
here = os.path.dirname(os.path.abspath(__file__))

# load index and all data
arrPar = os.path.join(here, prefix + '/parTraining.npy')
# arrPar = os.path.join(here, prefix + '/allData.npy')
arrPar = np.load(arrPar)


def find_duplicates(userIds, movieIds, ratings, movie1, movie2, overlap_arr):
    """assumes that userIds are sorted
    
    Arguments:
        userIds {[type]} -- [description]
        movieIds {[type]} -- [description]
        ratings {[type]} -- [description]
        movie1 {[type]} -- [description]
        movie2 {[type]} -- [description]
        overlap_arr {[type]} -- [description]
    """

    curr_ind, curr_user, found_match = 0, userIds[0], False
    found_movie1, found_movie2 = False, False 

    for uId, mId, r in zip(userIds, movieIds, ratings):
        if uId != curr_user:
            curr_user = uId 
            # reset
            found_movie1, found_movie2, found_match = False, False, False 

        # the user has rated both movie1 and movie2, we can skip until we reach the next user
        if found_match: 
            continue 

        if mId == movie1:
            found_movie1, overlap_arr[0, curr_ind]  = True, r 
        elif mId == movie2:
            found_movie2, overlap_arr[1, curr_ind] = True, r  
        else:
            continue 

        # reset 
        if found_movie1 and found_movie2:
            found_match  = True
            curr_ind += 1

    # set the limits of overlap_arr; beyond the -1s, there is no overlap
    overlap_arr[0, curr_ind], overlap_arr[1, curr_ind] = -1, -1


@numba.jit('void(uint32[:],uint16[:], int8[:], u8, u8, int8[:,:])', nopython = True)
def find_duplicates_jit(userIds, movieIds, ratings, movie1, movie2, overlap_arr):
    """assumes that userIds are sorted
    
    Arguments:
        userIds {[type]} -- [description]
        movieIds {[type]} -- [description]
        ratings {[type]} -- [description]
        movie1 {[type]} -- [description]
        movie2 {[type]} -- [description]
        overlap_arr {[type]} -- [description]
    """

    curr_ind, curr_user, found_match = 0, userIds[0], False
    found_movie1, found_movie2 = False, False 

    for uId, mId, r in zip(userIds, movieIds, ratings):
        if uId != curr_user:
            curr_user = uId 
            # reset
            found_movie1, found_movie2, found_match = False, False, False 

        # the user has rated both movie1 and movie2, we can skip until we reach the next user
        if found_match: 
            continue 

        if mId == movie1:
            found_movie1, overlap_arr[0, curr_ind]  = True, r 
        elif mId == movie2:
            found_movie2, overlap_arr[1, curr_ind] = True, r  
        else:
            continue 

        # reset 
        if found_movie1 and found_movie2:
            found_match  = True
            curr_ind += 1

    # set the limits of overlap_arr; beyond the -1s, there is no overlap
    overlap_arr[0, curr_ind], overlap_arr[1, curr_ind] = -1, -1


overlap_arr = np.zeros((2, num_users), dtype = 'int8')
movie1, movie2 = 23, 25
# find_duplicates(arrPar['userid'], arrPar['movieid'], arrPar['rating'], 23, 25, overlap_arr)

# find_duplicates_jit(arrPar['userid'], arrPar['movieid'], arrPar['rating'], 23, 25, overlap_arr)


# corrs2 = np.zeros((num_movies, num_movies), dtype = 'f4')
# calc_pearson_jit(arrPar['userid'], arrPar['movieid'], arrPar['rating'], num_movies, corrs2)

# diff_corr = corrs1 - corrs2
# print(np.amax(diff_corr))
# print(np.amax(corrs1)) 

# numTimesP = 3
# timetaken = timeit.Timer(partial(find_duplicates, arrPar['userid'], arrPar['movieid'], arrPar['rating'], movie1, movie2, overlap_arr)).timeit(number=numTimesP)
# print(timetaken)

numTimesN = 10
timetaken = timeit.Timer(partial(find_duplicates_jit, arrPar['userid'], arrPar['movieid'], arrPar['rating'], movie1, movie2, overlap_arr)).timeit(number=numTimesN)
print(timetaken)

