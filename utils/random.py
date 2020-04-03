from operator import mul
import numpy as np
from functools import reduce

def truncated_normal(*size, threshold=2):
	from scipy.stats import truncnorm
	all_size = reduce(mul, size, 1)
	values = truncnorm.rvs(-threshold, threshold, size=all_size)
	return np.reshape(values, size)
