import numba
import numpy as np

@numba.njit
def logr(x):
    if x > 0.0:
        return np.log(x)
    else:
        return -1000.0
