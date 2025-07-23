import numpy as np
import pandas as pd

from scipy.stats import wasserstein_distance
from scipy.spatial import distance
from scipy.spatial.distance import cdist

from pyemd import emd

def compute_distance_matrix(x=30, y=30):

	x_coords, y_coords = np.meshgrid(np.arange(x), np.arange(y))

	# Stack the coordinates into a single array
	coordinates = np.vstack((x_coords.ravel(), y_coords.ravel())).T
	
	# Cost matrix
	cost_matrix = cdist(coordinates, coordinates)

	return cost_matrix

# Reference -- code from: https://github.com/matthias-k/pysaliency/blob/master/pysaliency/metrics.py
def convert_saliency_map_to_density(saliency_map, minimum_value=0.0):
	if saliency_map.min() < 0:
		saliency_map = saliency_map - saliency_map.min()

	saliency_map = saliency_map + minimum_value

	saliency_map_sum = saliency_map.sum()
	
	if saliency_map_sum:
		saliency_map = saliency_map / saliency_map_sum
	else:
		saliency_map[:] = 1.0
		saliency_map /= saliency_map.sum()

	return saliency_map

def jensen_shannon_divergence(saliency_map_1, saliency_map_2):

	density_1 = convert_saliency_map_to_density(saliency_map_1)
	density_2 = convert_saliency_map_to_density(saliency_map_2)

	return 1 - distance.jensenshannon(density_1.flatten(), density_2.flatten(), base=2.0)

def CC(saliency_map_1, saliency_map_2):
	def normalize(saliency_map):
		saliency_map -= saliency_map.mean()
		std = saliency_map.std()

		if std:
			saliency_map /= std

		return saliency_map, std == 0

	smap1, constant1 = normalize(saliency_map_1.copy())
	smap2, constant2 = normalize(saliency_map_2.copy())

	if constant1 and not constant2:
		return 0.0
	else:
		return np.corrcoef(smap1.flatten(), smap2.flatten())[0, 1]

def KL_divergence(saliency_map_1, saliency_map_2):

	density_1 = convert_saliency_map_to_density(saliency_map_1).flatten() 
	density_2 = convert_saliency_map_to_density(saliency_map_2).flatten()

	eps = 1e-20 

	score = np.sum(density_1 * np.log(eps + density_1 / (density_2 + eps)))

	return score

def SIM(saliency_map_1, saliency_map_2):
	""" Compute similiarity metric. """
	density_1 = convert_saliency_map_to_density(saliency_map_1, minimum_value=0)
	density_2 = convert_saliency_map_to_density(saliency_map_2, minimum_value=0)

	return np.min([density_1, density_2], axis=0).sum()

# EMD expects the input to be observations
# https://stackoverflow.com/questions/57402842/why-is-the-wasserstein-distance-between-0-1-and-1-0-zero
# Earth Mover's Distance in 2D spaces   
def EMD(saliency_map_1, saliency_map_2, cost_matrix):

	density_1 = convert_saliency_map_to_density(saliency_map_1, minimum_value=0)
	density_2 = convert_saliency_map_to_density(saliency_map_2, minimum_value=0)

	return emd(density_1.flatten(), density_2.flatten(), cost_matrix)