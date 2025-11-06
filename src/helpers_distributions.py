

from scipy.stats import chi2
from scipy.stats import norm, gamma, multivariate_normal
# from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines

import math
import numpy as np
import P as P
import random


def _normal(X, mean, var, y_range):
	# Y = norm.pdf(X, loc=len(X)//2, scale=10)
	Y = norm.pdf(X, loc=mean, scale=var)
	Y = min_max_normalization(Y, y_range)
	return Y


def _gamma(X, mean, var, y_range):
	Y = gamma.pdf(X, mean, 0, var)
	Y = min_max_normalization(Y, y_range=y_range)
	return Y


def _log(X, y_range):
	Y = np.log(X)
	Y = min_max_normalization(Y, y_range=y_range)
	return Y


def _log_and_linear(X, y_range):  # hardcoded for now since only used for smoka
	Y = 0.99 * np.log(X) + 0.01 * X
	Y = min_max_normalization(Y, y_range=y_range)
	return Y


def min_max_normalization(X, y_range):

	"""

	"""

	new_min = y_range[0]
	new_max = y_range[1]
	Y = np.zeros(X.shape)

	_min = np.min(X)
	_max = np.max(X)

	for i, x in enumerate(X):
		Y[i] = ((x - _min) / (_max - _min)) * (new_max - new_min) + new_min

	return Y


def min_max_normalize_array(X, y_range):
	# # Normalize the values in the array to be between 0 and 1
	arr_min = X.min()
	arr_max = X.max()
	X_m = (X - arr_min) / (arr_max - arr_min)
	# modified_arr = normalized_arr * (new_max - new_min) + new_min
	X_m = X_m * (y_range[1] - y_range[0]) + y_range[0]

	return X_m

# def sigmoid(X):
# 	return 1/(1 + np.exp(-X))


def _sigmoid(x, grad_magn_inv=None, x_shift=None, y_magn=None, y_shift=None):
	"""
	the leftmost dictates gradient: 75=steep, 250=not steep
	the rightmost one dictates y: 0.1=10, 0.05=20, 0.01=100, 0.005=200
	y_magn???
	"""
	return (1 / (math.exp(-x / grad_magn_inv + x_shift) + y_magn)) + y_shift  # finfin


def sigmoid_blend(x, sharpness=5):
	"""The more sharpness, the crazier shift in speed possible"""
	return 1 / (1 + np.exp(-sharpness * (x - 0.5)))


def smoothstep(x, edge0=0, edge1=1):
	"""
	Smoothly interpolates between 0 and 1 as x transitions from edge0 to edge1.
	SLOPE ZERO AT BOTH ENDS

	Parameters
	----------
	x : float or ndarray
		Input value(s) to be mapped to a smooth 0–1 range.
	edge0 : float
		Lower transition edge. Values ≤ edge0 yield 0.
	edge1 : float
		Upper transition edge. Values ≥ edge1 yield 1.

	Returns
	-------
	float or ndarray
		A smoothstep-scaled value between 0 and 1.

	The 'edges' define the input range over which the smooth transition occurs.
	Values below edge0 output 0, above edge1 output 1, and in between produce
	a smooth cubic interpolation with zero slope at both ends.
	"""
	x = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
	return x * x * (3 - 2 * x)  # classic smoothstep


def sin_exp_experiment(X):
	"""
	Extension of cph. Firing frames is a combination of a number of normal distributions with specified nums and
	means in firing_info.
	"""

	cycles_currently = P.FRAMES_TOT / (2 * np.pi)
	# d = cycles_currently / P.EXPL_CYCLES  # divisor_to_achieve_cycles
	d = cycles_currently / P.EXPL_CYCLES  # divisor_to_achieve_cycles

	f_0 = 0.2  # firing prob coefficients
	f_1 = 0.05
	f_2 = 0.4

	left_shift = random.randint(int(-P.FRAMES_TOT), 0)
	Y = (f_0 * np.sin((X + left_shift) / d) +  # fast cycles
	     f_1 * np.sin((X + left_shift - 0) / (3 * d)) +  # slow cycles
	     f_2 * np.log((X + 10) / d) / np.log(P.FRAMES_TOT)) - 0.1  # prob of firing
	# Y = np.clip(Y, 0.0, 0.8)

	# Y = (2 * np.sin(X) + 0.5 * np.sin(X) + 0.4 * np.log(X) / np.log(len(X))) - 0.1

	return Y


def temperature_for_optimization(X):
	"""
	X is a list of
	"""
	T = np.zeros((2000,))
	t_cur = 10000
	d = 0.996
	for i in range(2000):
		T[i] = t_cur
		t_cur = t_cur * d

	delta = 200
	Y = np.exp(-abs(delta) / T)

	return Y


# def _multivariate_normal():
#
# 	rv = multivariate_normal(mean=[9, 9], cov=[[20, 0], [0, 20]])
#
# 	return rv

