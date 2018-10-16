import numpy as np 
import numba
import os

# user defined functions
import calc_correlation_matrix as corr
import calc_correlation_matrix_slow as corr_slow

# for debugging
import timeit
from sys import getsizeof
from functools import partial
test_times = True

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


@numba.jit('void(uint32[:],int32[:])', nopython = True)
def get_start_end_users(ids, ends):
    """takes an array of user ids that is sorted, and then finds the index at which they end
    
    Arguments:
        ids {uint32} -- user ids array
        ends {int32} -- the end index (in the training data) for each user
    """
    currUser = ids[0]
    # TODO: Can speed up with binary search!!!
    for ind, user in enumerate(ids):
        if user != currUser: 
            ends[currUser-1] = ind
            currUser = user
    
    # update the last entry
    ends[-1] = len(ids)

@numba.jit('void(uint16[:],int32[:])', nopython = True)
def get_start_end_movs(ids, ends):
    """takes an array of movie ids that is sorted, and then finds the index at which they end
    
    Arguments:
        ids {uint16} -- movie ids
        ends {int32} -- the end index (in the training data) for each movie
    """
    currMov = ids[0]
    # TODO: Can speed up with binary search!!!
    for ind, mov in enumerate(ids):
        if mov != currMov: 
            ends[currMov-1] = ind
            currMov = mov

    # update the last entry
    ends[-1] = len(ids)

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


################ massaging the arrays ###########################
ends_users = np.zeros(num_users, dtype = 'int32')
get_start_end_users(trainDataUM['userid'], ends_users)
ends_movs = np.zeros(num_movies, dtype = 'int32')
get_start_end_movs(trainDataMU['movieid'], ends_movs)

# test = ends_movs[1:]-ends_movs[:-1]
# (test == 20).nonzero()
# movie 9030 so index 9029 has 20 users rating it; same for index 11394
# movie 300 so index 299 has almost 150000 users rating it


############### learn a few about things about the data #############
most_ratings_by_any_user = most_ratings(ends_users)
most_ratings_of_any_mov = most_ratings(ends_movs)

############## calculating the overlap ##########################
range_ints = np.arange(num_data, dtype='uint32')
toFill = np.zeros((num_movies, 5, 5), dtype='uint32')
correlations = np.zeros((num_movies, num_movies), dtype='float64') 
corr.calc_corr(ends_movs, ends_users, trainDataMU['userid'], trainDataUM['movieid'], trainDataMU['rating'], trainDataUM['rating'], toFill, range_ints, correlations)

############ Compare with slow method; they should match!! ###########
# Some notes: 
# try edge cases: mov = 1 and num_movies
mov, range_movs = num_movies, np.arange(1, 1+num_movies)
correlations_slow = corr_slow.get_overlap_slow(mov, ends_movs, trainDataMU['userid'], trainDataMU['rating'], range_movs)

# get_overlap_1to1(1, 290, ends_movs, trainDataMU['userid'], trainDataMU['rating'])

test = correlations[mov-1]
print(correlations_slow[correlations_slow > -10][:10])
print(test[test>-10][:10])
test2 = correlations_slow-test
print("differences greater than 0.001")
print(test2[test2>0.001][:10])
print((test2>0.001).nonzero())


################### Test the speeds of different functions #############
if test_times:
    # for debugging
    print("Time taken by get_start_end_users")
    numTimesN = 1
    ends_users = np.zeros(num_users, dtype = 'int32')
    timetaken = timeit.Timer(partial(get_start_end_users, trainDataUM['userid'], ends_users)).timeit(number=numTimesN)
    print(timetaken)

    # for debugging
    print("Time taken by corr.calc_corr")
    numTimesN = 1
    toFill.fill(0)
    correlations.fill(0)
    timetaken = timeit.Timer(partial(corr.calc_corr, ends_movs, ends_users, trainDataMU['userid'], trainDataUM['movieid'], trainDataMU['rating'], trainDataUM['rating'], toFill, range_ints, correlations)).timeit(number=numTimesN)
    print(timetaken)

    # timing 
    mov, range_movs = 44, np.arange(1, 1+num_movies)
    print("Time taken by corr_slow.calc_corr")
    numTimesN = 1
    timetaken = timeit.Timer(partial(corr_slow.get_overlap_slow, mov, ends_movs, trainDataMU['userid'], trainDataMU['rating'], range_movs)).timeit(number=numTimesN) 
    print(timetaken)