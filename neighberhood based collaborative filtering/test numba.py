import numpy as np 
import numba
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

@numba.jit('void(u8, int32[:], int32[:], uint32[:],uint16[:], int8[:], int8[:], uint32[:,:,:], uint32[:])', nopython = True, parallel = True)
def get_overlap(m_index, ends_movs, ends_users, userIds, movieIds, r_mSort, r_uSort, toFill, range_ints): 
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
        range_ints {uint32[:]} -- an np array of integers from 0 to num_users ... it is passed for speedup reasons: we want to avoid all python
    """
    
    # find the users who rated movie m_index
    if m_index == 1:
        m_index_start = 0
    else:
        m_index_start = ends_movs[m_index-2]
    m_index_end = ends_movs[m_index-1]

    # for ind_u in range_ints[m_index_start:m_index_end]:
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
        # for ind_m in range_ints[user_id_start:user_id_end]:
        for ind_m in numba.prange(user_id_start, user_id_end):
            movie_id = movieIds[ind_m] 
            # the rating that user user_id gave to movie movie_id
            movie_id_rating = r_uSort[ind_m]

            # update toFill
            toFill[movie_id-1, m_index_rating-1, movie_id_rating-1] += 1

@numba.jit('void(u8, int32[:], int32[:], uint32[:],uint16[:], int8[:], int8[:], uint32[:,:,:], uint32[:], float32[:])', nopython = True)
def calc_pearson_1mov(m_index, ends_movs, ends_users, userIds, movieIds, r_mSort, r_uSort, toFill, range_ints, correlations): 
    """calc pearson corr
    
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
    get_overlap(m_index, ends_movs, ends_users, userIds, movieIds, r_mSort, r_uSort, toFill, range_ints)

    # now calc pearson corr 
    # TODO: change 17770 to num_movies!!
    for ind_mov in range_ints[:17770]:
        mat_ratings = toFill[ind_mov] 

        # first calculate the means
        
        # calculate the means of each movie, and how many users have rated both movie ind_mov and movie m_index
        mean_1, mean_2, num_overlaps = 0.0, 0.0, 0
        for i in range_ints[0:5]:
            for j in range_ints[0:5]:
                mean_1 +=  (i+1) * mat_ratings[i, j]
                mean_2 += (j+1) * mat_ratings[i, j]
                num_overlaps += mat_ratings[i, j]
        
        if num_overlaps > 1:
            mean_1 /= num_overlaps
            mean_2 /= num_overlaps

            # calculate the pearson correlation function
            num, den1, den2 = 0.0, 0.0, 0.0
            for i in range_ints[0:5]:
                for j in range_ints[0:5]:
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


@numba.jit('void(int32[:], int32[:], uint32[:],uint16[:], int8[:], int8[:], uint32[:,:,:], uint32[:], float32[:,:])', nopython = True)
def calc_pearson(ends_movs, ends_users, userIds, movieIds, r_mSort, r_uSort, toFill, range_ints, correlations):
    # TODO: replace 17771 with 1 + num_movies !!
    for mov in range_ints[1:100]:
        toFill.fill(0) #initialize to 0

        calc_pearson_1mov(mov, ends_movs, ends_users, userIds, movieIds, r_mSort, r_uSort, toFill, range_ints, correlations[mov-1])

# @numba.jit('void(u8, u8, int32[:], uint32[:], int8[:])')  
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
toFill = np.zeros((num_movies, 5, 5), dtype='uint32')
correlations = np.zeros(num_movies, dtype='float32') 

range_ints = np.arange(num_data, dtype='uint32')
m_index = 1
# calc_pearson_1mov(m_index, ends_movs, ends_users, trainDataMU['userid'], trainDataUM['movieid'], trainDataMU['rating'], trainDataUM['rating'], toFill, range_ints, correlations)

# TODO: reset toFill after each usage

# for debugging
numTimesN = 1
timetaken = timeit.Timer(partial(calc_pearson_1mov, m_index, ends_movs, ends_users, trainDataMU['userid'], trainDataUM['movieid'], trainDataMU['rating'], trainDataUM['rating'], toFill, range_ints, correlations)).timeit(number=numTimesN)
print(timetaken)

toFill = np.zeros((num_movies, 5, 5), dtype='uint32')
correlations = np.zeros((num_movies, num_movies), dtype='float32') 
# for debugging
numTimesN = 1
timetaken = timeit.Timer(partial(calc_pearson, ends_movs, ends_users, trainDataMU['userid'], trainDataUM['movieid'], trainDataMU['rating'], trainDataUM['rating'], toFill, range_ints, correlations)).timeit(number=numTimesN)
print(timetaken)

mov, range_movs = 1, np.arange(1, 1+num_movies)
correlations_slow = get_overlap_slow(mov, ends_movs, trainDataMU['userid'], trainDataMU['rating'], range_movs)
# numTimesN = 1
# timetaken = timeit.Timer(partial(get_overlap_slow, mov, ends_movs, trainDataMU['userid'], trainDataMU['rating'], range_movs)).timeit(number=numTimesN) 
# print(timetaken)


############### for calculating overlap ##########################
overlap_arr = np.zeros((2, num_users), dtype = 'int8')
movie1, movie2 = 23, 25

# numTimesN = 10
# timetaken = timeit.Timer(partial(find_duplicates_jit, trainDataUM['userid'], trainDataUM['movieid'], trainDataUM['rating'], movie1, movie2, overlap_arr)).timeit(number=numTimesN)
# print(timetaken)

