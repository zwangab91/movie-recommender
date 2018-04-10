# playing around 

import numpy as np 
import numba
import os

# for debugging
import timeit
from sys import getsizeof
from functools import partial

num_movies, num_users = 17770, 458293
# location of the current file
here = os.path.dirname(os.path.abspath(__file__))

# load um data
prefix = '../data/um'
trainDataUM = os.path.join(here, prefix + '/parTraining.npy')
trainDataUM = os.path.join(here, prefix + '/allData.npy')
trainDataUM = np.load(trainDataUM)
num_data = len(trainDataUM)
# load mu data
prefix = '../data/mu'
trainDataMU = os.path.join(here, prefix + '/allData.npy')
trainDataMU = np.load(trainDataMU)


@numba.jit('void(uint32[:],int32[:])', nopython = True)
def get_start_end_users(ids, ends):
    """takes an array of user ids that is sorted, and then finds the index at which they end
    
    Arguments:
        ids {uint32} -- user ids array
        ends {int32} -- the end index (in the training data) for each user
    """
    currId = ids[0]
    # TODO: Can speed up with binary search!!!
    for ind, iden in enumerate(ids):
        if iden != currId: 
            ends[currId-1] = ind
            currId = iden

@numba.jit('void(uint16[:],int32[:])', nopython = True)
def get_start_end_movs(ids, ends):
    """takes an array of movie ids that is sorted, and then finds the index at which they end
    
    Arguments:
        ids {uint16} -- movie ids
        ends {int32} -- the end index (in the training data) for each movie
    """
    currId = ids[0]
    # TODO: Can speed up with binary search!!!
    for ind, iden in enumerate(ids):
        if iden != currId: 
            ends[currId-1] = ind
            currId = iden

@numba.jit('u8(int32[:])', nopython = True)
def most_ratings(ends):
    """the largest number of ratings an object has associated with (e.g. the most ratings any movie has)
    
    Arguments:
        ends {uint16} -- end indices of the item in their associated array
    Returns:
        largest {u8} -- will be updated with the largest number of ratings an object has associated with (e.g. the most ratings any movie has)
    """
    largest, prev_end = 0, 0
    
    for e in ends:
        largest = max(largest, e-prev_end)
        prev_end = e

    return largest

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
    if m_index == 1:
        m_index_start = 0
    else:
        m_index_start = ends_movs[m_index-2]
    m_index_end = ends_movs[m_index-1]

    # find the users that rated movie m_index
    for ind_u, user_id in enumerate(userIds[m_index_start:m_index_end], m_index_start):
        # the rating that user user_id gave to movie m_index
        m_index_rating = r_mSort[ind_u]
        #find the movies rated by each user user_id
        if user_id == 1:
            user_id_start = 0
        else:
            user_id_start = ends_users[user_id-2]
        user_id_end = ends_movs[user_id-1]
        for ind_m, movie_id in enumerate(movieIds[user_id_start:user_id_end], user_id_start):
            # the rating that user user_id gave to movie movie_id
            movie_id_rating = r_uSort[ind_m]

            # update toFill
            toFill[movie_id-1, m_index_rating-1, m_index_rating-1]
            

    return 1

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

################ massaging the arrays ###########################
ends_users = np.zeros(num_users, dtype = 'int32')
get_start_end_users(trainDataUM['userid'], ends_users)
ends_movs = np.zeros(num_users, dtype = 'int32')
get_start_end_movs(trainDataMU['movieid'], ends_movs)

# # for debugging
# numTimesN = 10
# timetaken = timeit.Timer(partial(get_start_end_users, trainDataUM['userid'], ends)).timeit(number=numTimesN)
# print(timetaken)

############### learn a few about things about the data #############
most_ratings_by_any_user = most_ratings(ends_users)
most_ratings_of_any_mov = most_ratings(ends_movs)

############## calculating the overlap ##########################


############### for calculating overlap ##########################
overlap_arr = np.zeros((2, num_users), dtype = 'int8')
movie1, movie2 = 23, 25

# numTimesN = 10
# timetaken = timeit.Timer(partial(find_duplicates_jit, trainDataUM['userid'], trainDataUM['movieid'], trainDataUM['rating'], movie1, movie2, overlap_arr)).timeit(number=numTimesN)
# print(timetaken)

