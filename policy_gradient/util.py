from gym.spaces import Box, Discrete
import numpy as np
from scipy.signal import lfilter

def flatten_space(space):
	if isinstance(space, Box):
		return np.prod(space.shape)
	elif isinstance(space, Discrete):
		return space.n
	else:
		raise ValueError("Env must be either Box or Discrete.")

def discount_cumsum(x, discount_rate):
    return lfilter([1], [1, -discount_rate], x[::-1], axis=0)[::-1]