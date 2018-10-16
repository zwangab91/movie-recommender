import numpy as np

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

    for ind_u in range_ints[m_index_start:m_index_end]:
        user_id = userIds[ind_u]
        # the rating that user user_id gave to movie m_index
        m_index_rating = r_mSort[ind_u]
        #find the movies rated by each user user_id
        if user_id == 1:
            user_id_start = 0
        else:
            user_id_start = ends_users[user_id-2]
        user_id_end = ends_movs[user_id-1]
        for ind_m in range_ints[user_id_start:user_id_end]:
            movie_id = movieIds[ind_m] 
            # the rating that user user_id gave to movie movie_id
            movie_id_rating = r_uSort[ind_m]

            # update toFill
            toFill[movie_id-1, m_index_rating-1, movie_id_rating-1] += 1

