import numpy as np

def cum_sum_until_with_numpy(f, k):
    sum = np.cumsum(f)
    index = np.searchsorted(sum, k, side='right')
    return index, sum[index-1], sum[index]

def cum_sum_check(f, k):
    index = 0
    csum = f[index]
    while csum < k:
        index = index + 1
        csum += f[index]

    f_index = csum
    f_index_minus_1 = csum - f[index]

    return index, f_index_minus_1, f_index