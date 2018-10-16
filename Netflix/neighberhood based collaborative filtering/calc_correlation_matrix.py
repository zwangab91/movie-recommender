import numpy as np 
import numba
import os
import math

# for debugging
import timeit
from sys import getsizeof
from functools import partial

@numba.jit('void(u8, int32[:], int32[:], uint32[:],uint16[:], int8[:], int8[:], uint32[:,:,:])', nopython = True, parallel = False)
def get_overlap(m_index, ends_movs, ends_users, userIds, movieIds, r_mSort, r_uSort, toFill): 
    """for movie with index m_index, find users that have rated m_index and another movie. toFill will be updated with how users rate each differently
    
    Arguments:
        m_index {uint16} -- the movie that will be compared to the other movies 
        ends_movs {int32[:]} -- end indices of the movies
        ends_users {int32[:]} -- end indices of the users
        userIds {uint32[:]} -- the user ids (sorted originally by movies)
        movieIds {uint16[:]} -- the movie ids (sorted originally by users)
        r_mSort {int8[:]} -- the ratings associated with the user ids (so assuming originally sorted by movies)
        r_uSort {int8[:]} -- the ratings associated with the movie ids (so assuming originally sorted by users)
        toFill {int32[:]} -- toFill[i,j,k] is how many users rated movie_m_index (j+1) stars and rated movie_(i+1) (k+1) stars
    """
    
    # find the users who rated movie m_index
    if m_index == 1:
        m_index_start = 0
    else:
        m_index_start = ends_movs[m_index-2]
    m_index_end = ends_movs[m_index-1]

    for ind_u in numba.prange(m_index_start, m_index_end):
        user_id = userIds[ind_u]
        # the rating that user user_id gave to movie m_index
        m_index_rating = r_mSort[ind_u]
        #find the movies rated by user user_id
        if user_id == 1:
            user_id_start = 0
        else:
            user_id_start = ends_users[user_id-2]
        user_id_end = ends_users[user_id-1]
        # fill toFill
        for ind_m in numba.prange(user_id_start, user_id_end):
            movie_id = movieIds[ind_m] 
            # the rating that user user_id gave to movie movie_id
            movie_id_rating = r_uSort[ind_m]

            # update toFill
            toFill[movie_id-1, m_index_rating-1, movie_id_rating-1] += 1

@numba.jit('void(u8, int32[:], int32[:], uint32[:],uint16[:], int8[:], int8[:], uint32[:,:,:], uint32[:], float64[:])', nopython = True, parallel = False)
def calc_pearson_1mov(m_index, ends_movs, ends_users, userIds, movieIds, r_mSort, r_uSort, toFill, range_ints, correlations): 
    """calc pearson corr of movie m_index with all the other movies. Correlations will hold the values of these correlations
    
    Arguments:
        m_index {u8} -- the movie that will be compared to the other movies 
        ends_movs {int32[:]} -- end indices of the movies
        ends_users {int32[:]} -- end indices of the users
        userIds {uint32[:]} -- the user ids (sorted originally by movies)
        movieIds {uint16[:]} -- the movie ids (sorted originally by users)
        r_mSort {int8[:]} -- the ratings associated with the user ids (so assuming originally sorted by movies)
        r_uSort {int8[:]} -- the ratings associated with the movie ids (so assuming originally sorted by users)
        toFill {int32[:]} -- toFill[i,j,k] is how many users rated movie_m_index (j+1) stars and rated movie_(i+1) (k+1) stars
        range_ints {uint32[:]} -- an np array of integers from 0 to num_users ... it is passed for speedup reasons: we want to avoid all python
        correlations {float32[:]} -- correlations of the movie m_index with all the other movies
    """
    # update toFill with ratings
    get_overlap(m_index, ends_movs, ends_users, userIds, movieIds, r_mSort, r_uSort, toFill)

    num_movies = correlations.shape[0]
    # now calc pearson corr 
    # for ind_mov in range_ints[:num_movies]:
    for ind_mov in numba.prange(num_movies):
        mat_ratings = toFill[ind_mov] 

        # first calculate the means
        
        # calculate the means of each movie, and how many users have rated both movie ind_mov and movie m_index
        mean_1, mean_2, num_overlaps = 0.0, 0.0, 0
        for i in numba.prange(5):
            for j in numba.prange(5):
                mean_1 +=  (i+1) * mat_ratings[i, j]
                mean_2 += (j+1) * mat_ratings[i, j]
                num_overlaps += mat_ratings[i, j]
        
        if num_overlaps > 1:
            mean_1 /= num_overlaps
            mean_2 /= num_overlaps

            # calculate the pearson correlation function
            num, den1, den2 = 0.0, 0.0, 0.0 
            for i in numba.prange(5):
                for j in numba.prange(5):
                    num_repeats = mat_ratings[i, j]
                    r_1 = i+1 #rating of movie 1
                    r_2 = j+1 #rating of movie 2
                    num += num_repeats * (r_1-mean_1)*(r_2-mean_2)
                    den1 += num_repeats * (r_1-mean_1)**2
                    den2 += num_repeats * (r_2-mean_2)**2
            if abs(den1) > 1e-10 and abs(den2) > 1e-10:  
                correlations[ind_mov] = num/(den1**0.5)/(den2**0.5)
            else:
                correlations[ind_mov] = -10    
        else:
            correlations[ind_mov] = -10


@numba.jit('void(int32[:], int32[:], uint32[:],uint16[:], int8[:], int8[:], uint32[:,:,:], uint32[:], float64[:,:])', nopython = True, parallel = False)
def calc_corr(ends_movs, ends_users, userIds, movieIds, r_mSort, r_uSort, toFill, range_ints, correlations):
    """calc corrrelation matrix of all the movies
    
    Arguments:
        ends_movs {int32[:]} -- end indices of the movies
        ends_users {int32[:]} -- end indices of the users
        userIds {uint32[:]} -- the user ids (sorted originally by movies)
        movieIds {uint16[:]} -- the movie ids (sorted originally by users)
        r_mSort {int8[:]} -- the ratings associated with the user ids (so assuming originally sorted by movies)
        r_uSort {int8[:]} -- the ratings associated with the movie ids (so assuming originally sorted by users)
        toFill {int32[:]} -- toFill[i,j,k] is how many users rated movie_m_index (j+1) stars and rated movie_(i+1) (k+1) stars
        range_ints {uint32[:]} -- an np array of integers from 0 to num_users ... it is passed for speedup reasons: we want to avoid all python
        correlations {float64[:, :]} -- correlations of the movie m_index with all the other movies
    """
    num_movies = correlations.shape[0]
    for mov in numba.prange(1,1+num_movies):
    # for mov in numba.prange(1,500):
        toFill.fill(0) #initialize to 0

        calc_pearson_1mov(mov, ends_movs, ends_users, userIds, movieIds, r_mSort, r_uSort, toFill, range_ints, correlations[mov-1])