from scipy.ndimage import *
from quick_filter import quick_filter
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

mode = 'wrap'
cval = 0.0

def compare_times(n, m, percent=0.5):

    signal = np.random.randn(n)

    start = time.time()
    median_filter(signal, size=m, mode=mode, cval=cval)
    end = time.time()
    scipy_time = end-start

    start = time.time()
    quick_filter(signal, percent=percent, window_size=m, truncate_mode='same', edge_mode=mode, cval=cval)
    end = time.time()
    my_time = end-start

    return scipy_time/my_time

def compare_results(n, m, percent=0.5):

    signal = np.random.randn(n)

    scipy_attempt = median_filter(signal, size=m, mode=mode, cval=cval)
    my_attempt = quick_filter(signal, percent=percent, window_size=m, truncate_mode='same', edge_mode=mode, cval=cval)#[m//2:-m//2+1]

    diff = np.abs(scipy_attempt - my_attempt)
    # plt.plot(my_attempt * np.array([int(i>0) for i in diff]))
    # plt.plot(my_attempt, color='r')
    # plt.plot(scipy_attempt, color='b')
    # plt.show()
    return np.sum(diff)

print(compare_times(1<<12, 1<<12, percent=0.5))
print(compare_results(1<<12, 1<<12, percent=0.5))