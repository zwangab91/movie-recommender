# !!!!!!!!!!!!! IMPORTANT: DO NOT RUN THIS UNDER A DEBUGGER!!!!!!!! because it creates issues with loadtxt (np.resize in loadtxt would have an issue with the debugger referencing the same array that is being resized)

import numpy as np 
import os

def convertToNParr(prefix = 'um'):
    """convert the data text files provided by yasser to memory-efficient numpy arrays, that can be quickly loaded
    
    Keyword Arguments:
        prefix {str} -- [the prefix for which directory to use: either um or mu] (default: {'um'})
    """

    if prefix != 'um' and prefix != 'mu':
        raise ValueError("works for the um and mu directories only!") 

    
    # location of the current file
    here = os.path.dirname(os.path.abspath(__file__))

    # convert the index 
    idxfname = os.path.join(here, prefix + '/all.idx') # idx file location
    arr = np.loadtxt(idxfname, dtype = 'int8')
    savename = os.path.join(here, prefix + '/allIndices')
    np.save(savename, arr)
    print('done converting indices for ' + prefix)

    # convert the all.dta
    allData = os.path.join(here, prefix + '/all.dta')
    arr = np.loadtxt(allData, dtype={'names': ('userid', 'movieid', 'date', 'rating'), 'formats': ('uint32', 'uint16', 'uint16', 'int8')})
    savename = os.path.join(here, prefix + '/allData')
    np.save(savename, arr)
    print('done converting all data for ' + prefix)

    # convert the qual.dta
    qualData = os.path.join(here, prefix + '/qual.dta')
    arr = np.loadtxt(qualData, dtype={'names': ('userid', 'movieid', 'date'), 'formats': ('uint32', 'uint16', 'uint16')})
    savename = os.path.join(here, prefix + '/qualData')
    np.save(savename, arr)
    print('done converting qual data for ' + prefix)
    

    
convertToNParr('um')
convertToNParr('mu')