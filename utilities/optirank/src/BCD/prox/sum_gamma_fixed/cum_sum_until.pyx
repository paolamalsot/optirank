import numpy as np
cimport numpy as np
cimport cython

#@cython.boundscheck(False)

DTYPE = np.float64

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.float_t DTYPE_t

@cython.boundscheck(True) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function


def do_cumsum(np.ndarray[DTYPE_t, ndim=1] f, int k):
    """
    computes cum_sum_j = sum_i=0^j{f(i)} such that j is the smallest index for which cum_sum_j>k
    :param f: numpy array with float64
    :param k: integer
    :return: j, cum_sum_j, cum_sum_(j-1)
    Note that cum_sum_j > k and cum_sum_(j-1) <=k
    """
    cdef int index = 0
    cdef DTYPE_t csum_index
    cdef DTYPE_t csum_index_minus_1
    cdef DTYPE_t csum = f[index]
    cdef int size_array = len(f)

    while csum <= k:
        index = index + 1
        csum += f[index]

        if index == size_array - 1:
            break

    csum_index = csum
    csum_index_minus_1 = csum - f[index]

    return index, csum_index_minus_1, csum_index