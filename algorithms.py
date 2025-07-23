import numpy as np
import pandas as pd
import os
import matlab.engine
import math

from PIL import Image
from skimage.metrics import structural_similarity

from scipy.spatial import distance
from evaluation_metrics import *
from utils import *

import cv2
from scipy.spatial import ConvexHull

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors

from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

from scipy.spatial.distance import cdist

import time

# Store info about representative number of points C for AppropPaWs
number_of_reps = {

	'estate_corr': {'box_sampling_0.01_0.01': 2,
					'box_sampling_0.01_0.001': 2,
					'box_sampling_0.01_0.0001': 2,
					'box_sampling_0.005_0.01': 2,
					'box_sampling_0.005_0.001': 2,
					'box_sampling_0.005_0.0001': 2,
					'box_sampling_0.001_0.01': 2, 
					'box_sampling_0.001_0.001': 2,
					'box_sampling_0.001_0.0001': 2,
					'box_sampling_0.002_0.01': 2,
					'box_sampling_0.002_0.001': 2,
					'box_sampling_0.002_0.0001': 2,
					'box_sampling_0.003_0.01': 2,
					'box_sampling_0.003_0.001': 2,
					'box_sampling_0.003_0.0001': 2}, 

	'estate_anomalies': {'box_sampling_0.01_0.01': 2,
						 'box_sampling_0.01_0.001': 2,
						 'box_sampling_0.01_0.0001': 2,
						 'box_sampling_0.005_0.01': 2,
						 'box_sampling_0.005_0.001': 2,
						 'box_sampling_0.005_0.0001': 2,
						 'box_sampling_0.001_0.01': 2, 
						 'box_sampling_0.001_0.001': 2,
						 'box_sampling_0.001_0.0001':2,
						 'box_sampling_0.002_0.01': 2, 
						 'box_sampling_0.002_0.001': 2,
						 'box_sampling_0.002_0.0001':2,
						 'box_sampling_0.003_0.01': 2, 
						 'box_sampling_0.003_0.001': 2,
						 'box_sampling_0.003_0.0001':2,
						 },

	'epileptic_corr': {'box_sampling_0.01_0.01': 2,
						 'box_sampling_0.01_0.001': 2,
						 'box_sampling_0.01_0.0001': 2,
						 'box_sampling_0.005_0.01': 2,
						 'box_sampling_0.005_0.001': 2,
						 'box_sampling_0.005_0.0001': 2,
						 'box_sampling_0.001_0.01': 2, 
						 'box_sampling_0.001_0.001': 2,
						 'box_sampling_0.001_0.0001':2,
						 'box_sampling_0.002_0.01': 2, 
						 'box_sampling_0.002_0.001': 2,
						 'box_sampling_0.002_0.0001':2,
						 'box_sampling_0.003_0.01': 2, 
						 'box_sampling_0.003_0.001': 2,
						 'box_sampling_0.003_0.0001':2},

	'mnist': {'box_sampling_0.01_0.01': 5,
			  'box_sampling_0.01_0.001': 5,
			  'box_sampling_0.01_0.0001': 5,
			  'box_sampling_0.005_0.01': 5,
			  'box_sampling_0.005_0.001': 5,
			  'box_sampling_0.005_0.0001': 5,
			  'box_sampling_0.001_0.01': 5, 
			  'box_sampling_0.001_0.001': 2,
			  'box_sampling_0.001_0.0001': 2, 
			  'box_sampling_0.002_0.01': 5, 
			  'box_sampling_0.002_0.001': 5,
			  'box_sampling_0.002_0.0001': 5,
			  'box_sampling_0.003_0.01': 5, 
			  'box_sampling_0.003_0.001': 5,
			  'box_sampling_0.003_0.0001': 5}, 

	'hidden_corr': {'box_sampling_0.01_0.01': 5,
					'box_sampling_0.01_0.001': 5,
					'box_sampling_0.01_0.0001': 5,
					'box_sampling_0.005_0.01': 5,
					'box_sampling_0.005_0.001': 5,
					'box_sampling_0.005_0.0001': 5,
					'box_sampling_0.001_0.01': 5, 
					'box_sampling_0.001_0.001': 5,
					'box_sampling_0.001_0.0001': 5,
					'box_sampling_0.002_0.01': 5, 
					'box_sampling_0.002_0.001': 5,
					'box_sampling_0.002_0.0001': 5,
					'box_sampling_0.003_0.01': 5, 
					'box_sampling_0.003_0.001': 5,
					'box_sampling_0.003_0.0001': 5}, 

	'acsi': {'box_sampling_0.01_0.01': 5,
			 'box_sampling_0.01_0.001': 5,
			 'box_sampling_0.01_0.0001': 5,
			 'box_sampling_0.005_0.01': 5,
			 'box_sampling_0.005_0.001': 5,
			 'box_sampling_0.005_0.0001': 5,
			 'box_sampling_0.001_0.01': 5, 
			 'box_sampling_0.001_0.001': 5,
			 'box_sampling_0.001_0.0001':5}, 

	'intro': {'box_sampling_0.01_0.01': 7,
			  'box_sampling_0.005_0.001': 7,
			  'box_sampling_0.001_0.001': 7,
			  'box_sampling_0.005_0.01': 7,
			  'box_sampling_0.01_0.001': 7,
			  'box_sampling_0.001_0.0001':7}          
}

###########################################################
# Functions needed for Max-Min & PAwS
###########################################################
def dis_matrix(data_x, data_y, point_x, point_y):
	return np.sqrt(np.square(data_x-point_x)+np.square(data_y-point_y))

# Code to implement GMM #
def max_min_scalable_with_initial_points(data_x, data_y, initial_points, cur_seed=0, use_seed=False, sample=1):
	
	if use_seed:
		np.random.seed(cur_seed)
		
	indices = []
	
	# Initial set of points is empty
	if len(initial_points) == 0: 
		starting_index = np.random.randint(len(data_x)) #rnd(0,len(data_x)-1)
		indices = [starting_index]
		dist = dis_matrix(data_x, data_y, data_x[starting_index], data_y[starting_index])
		
	# Already selected some points
	else:
		
		dist = dis_matrix(data_x, data_y, initial_points[0,0], initial_points[0,1])

		for cc in range(1, len(initial_points)):
			dist = np.minimum(dist, dis_matrix(data_x, data_y, initial_points[cc, 0], initial_points[cc, 1]))
	
	while(len(indices) < sample):
		idx = np.argmax(dist)
		indices.append(idx)
		
		dist = np.minimum(dist, dis_matrix(data_x, data_y, data_x[idx], data_y[idx])) #* candidate 

	return np.array(indices)

# Code to implement PaWs #
def max_min_scalable_with_initial_points_with_saliency(data_x, data_y, data_z, initial_points, cur_seed=0, use_seed=False, sample=1):
	
	if use_seed:
		np.random.seed(cur_seed)
		
	indices = []
		
	comp_time = 0

	# Initial set of points is empty
	if len(initial_points) == 0: 
		starting_index = np.random.randint(len(data_x)) #rnd(0,len(data_x)-1)
		indices = [starting_index]
		dist = dis_matrix(data_x, data_y, data_x[starting_index], data_y[starting_index]) * data_z
		
	# Already selected some points
	else:
		
		dist = dis_matrix(data_x, data_y, initial_points[0,0], initial_points[0,1]) * data_z

		for cc in range(1, len(initial_points)):
			dist = np.minimum(dist, dis_matrix(data_x, data_y, initial_points[cc, 0], initial_points[cc, 1])*data_z)
	
	while(len(indices) < sample):
		idx = np.argmax(dist)
		indices.append(idx)

		start = time.time()
		dist = np.minimum(dist, dis_matrix(data_x, data_y, data_x[idx], data_y[idx])*data_z)
		end = time.time()
		
		comp_time += (end-start)

	return np.array(indices)

########################################################### 

class Sampler():

	def __init__(self, dataset, method_name, plotting_info):

		# Storage paths 
		self.dataset = dataset
		self.parent_path = self.dataset
		self.data_saliency_path = os.path.join(self.parent_path, 'saliency_data')
		self.sampling_path = os.path.join(self.parent_path, 'sampling_techniques')
		self.method_path = os.path.join(self.sampling_path, method_name)
		self.method_name = method_name

		# Information for plotting the samples information
		self.x_canvas_size = plotting_info['x_canvas_size']
		self.y_canvas_size = plotting_info['y_canvas_size']
		self.dpi_val = plotting_info['dpi_value']
		self.marker = plotting_info['marker']
		self.alpha = plotting_info['alpha']
		self.xlim = plotting_info['x_lim']
		self.ylim = plotting_info['y_lim']
		self.color = plotting_info['color']
		self.background = plotting_info['background_color']

		# Attributes for coming up with cumulative saliency
		self.transparency_levels = plotting_info['transparency_values'] #transparency_levels
		self.marker_levels = plotting_info['marker_values'] #marker_levels

		# Parameter for generating cumulative maps
		self.marker_level_start = plotting_info['marker_level_start']
		self.marker_level_end = plotting_info['marker_level_end']

		self.transparency_level_start = plotting_info['transparency_level_start']
		self.transparency_level_end = plotting_info['transparency_level_end']
		
		# Resize images to common canvas size because removing padding results to different sizes 
		self.resize_x =  plotting_info['resize_x']
		self.resize_y =  plotting_info['resize_y']

		# Path to store info for cumulative saliency maps 
		self.cumulative_images = os.path.join(self.method_path, 'cumulative_saliency/images')
		self.cumulative_data = os.path.join(self.method_path, 'cumulative_saliency/data')
		self.cumulative_maps = os.path.join(self.method_path, 'cumulative_saliency/maps')
		self.cumulative_metrics = os.path.join(self.method_path, 'cumulative_saliency')

		self.metrics_path = self.sampling_path 
		self.gamma_param = plotting_info['gamma']

		self.engine = matlab.engine.start_matlab()
	
	def create_storage_paths(self, seeds):

		if not os.path.exists(self.cumulative_metrics):
			os.makedirs(self.cumulative_metrics)

		if not os.path.exists(self.cumulative_data):
			os.makedirs(self.cumulative_data)

		if not os.path.exists(self.cumulative_maps):
			os.makedirs(self.cumulative_maps)

		if not os.path.exists(self.cumulative_images):
			os.makedirs(self.cumulative_images)

		# Generate subfolders to support multiple repetitions
		for seed in seeds:

			if not os.path.exists(os.path.join(self.cumulative_data, f'repetition_{seed}')):
				os.makedirs(os.path.join(self.cumulative_data, f'repetition_{seed}'))

			if not os.path.exists(os.path.join(self.cumulative_maps, f'repetition_{seed}')):
				os.makedirs(os.path.join(self.cumulative_maps, f'repetition_{seed}'))

			if not os.path.exists(os.path.join(self.cumulative_images, f'repetition_{seed}')):
				os.makedirs(os.path.join(self.cumulative_images, f'repetition_{seed}'))

	def store_metrics_summary(self, results, sal_type='cumulative'):

		df = pd.DataFrame(results, columns=['score', 'size', 'metric'])

		if sal_type == 'cumulative':
			df.to_csv(os.path.join(self.cumulative_metrics, f'{self.method_name}_metrics.csv'), index=False)

		elif sal_type == 'cumulative_simple':   
			df.to_csv(os.path.join(self.metrics_path, f'{self.method_name}_metrics.csv'), index=False)      
			
	def store_sample(self, data, sample_size, cur_seed):

		path_to_data = os.path.join(f'{self.cumulative_data}/repetition_{cur_seed}', f'{self.method_name}_{sample_size}.npz')
		np.savez(path_to_data, data=data)

	def plot_sample(self, sample_size):

		# Read the sample json file you want to plot
		path_to_data = os.path.join(self.cumulative_data, f'{self.method_name}_{sample_size}.npz')
		data_np = np.load(path_to_data)['data']

		storage_img_path = os.path.join(self.cumulative_images, f'{self.method_name}_{sample_size}')

		# Initial image
		custom_plot_function(data_np, self.xlim, self.ylim, storage_img_path, self.dpi_val, 
								background_color=self.background, my_color=self.color,
									marker_size=self.marker, t=self.alpha, x_axis_size=self.x_canvas_size, y_axis_size=self.y_canvas_size) 

	def generate_sample_saliency(self, sample_size, cur_seed):

		if not os.path.exists(os.path.self.cumulative_images):
			os.makedirs(storage_path)

		# Read the image for which you want to compute the saliency map 
		img = os.path.join(f'{self.cumulative_images}/repetition_{cur_seed}', f'{self.method_name}_{sample_size}.png')

		saliency_map_path = os.path.join(f'{self.cumulative_maps}/repetition_{cur_seed}', f'{self.method_name}_{sample_size}.npz')

		saliency_map = self.engine.generate_full_saliency(img)
		saliency_map = np.array(saliency_map)

		np.savez(saliency_map_path, data=saliency_map)

	def plot_sample_saliency(self, sample_size, cur_seed):

		saliency_map = np.load(os.path.join(f'{self.cumulative_maps}/repetition_{cur_seed}', f'{self.method_name}_{sample_size}.npz'))['data']

		saliency_img_path = os.path.join(f'{self.cumulative_images}/repetition_{cur_seed}', f'{self.method_name}_{sample_size}')

		custom_plot_function(saliency_map, self.xlim, self.ylim, saliency_img_path, self.dpi_val, plot_type='image', color_map=plt.cm.gray,
								background_color=self.background, my_color=self.color,
									marker_size=self.marker, t=self.alpha, x_axis_size=self.x_canvas_size, y_axis_size=self.y_canvas_size)

	# Function to plot all the intermediate instances for a fixed sample    
	def generate_intermediate_plots(self, sample_size, cur_seed):

		# Read the sample json file you want to plot
		path_to_data = os.path.join(f'{self.cumulative_data}/repetition_{cur_seed}', f'{self.method_name}_{sample_size}.npz')
		data_np = np.load(path_to_data)['data']

		for cur_marker in self.marker_levels:

			for alpha_value in self.transparency_levels:

				storage_img_path = os.path.join(f'{self.cumulative_images}/repetition_{cur_seed}', f'{self.method_name}_{sample_size}_alpha_{alpha_value}_marker_{cur_marker}')

				custom_plot_function(data_np, self.xlim, self.ylim, storage_img_path, self.dpi_val, 
										background_color=self.background, my_color=self.color,
											marker_size=cur_marker, t=alpha_value, x_axis_size=self.x_canvas_size, y_axis_size=self.y_canvas_size)

	# Function to generate all the intermediate saliency maps for a sample          
	def generate_intermediate_saliency_maps(self, sample_size, cur_seed):

		for cur_marker in self.marker_levels:

			for alpha_value in self.transparency_levels:

				img = os.path.join(f'{self.cumulative_images}/repetition_{cur_seed}', f'{self.method_name}_{sample_size}_alpha_{alpha_value}_marker_{cur_marker}.png')
				print("Computing saliency map:",  f'{self.method_name}_{sample_size}_alpha_{alpha_value}_marker_{cur_marker}.png')

				saliency_map_path = os.path.join(f'{self.cumulative_maps}/repetition_{cur_seed}', f'{self.method_name}_{sample_size}_alpha_{alpha_value}_marker_{cur_marker}.npz')
				saliency_map = self.engine.generate_full_saliency(img)
				saliency_map = np.array(saliency_map)

				np.savez(saliency_map_path, data=saliency_map)

	# Function to plot all the intermediate saliency maps for a sample          
	def plot_intermediate_saliency_maps(self, sample_size, cur_seed):

		for cur_marker in self.marker_levels:

			for alpha_value in self.transparency_levels:

				saliency_data = np.load(os.path.join(f'{self.cumulative_maps}/repetition_{cur_seed}', f'{self.method_name}_{sample_size}_alpha_{alpha_value}_marker_{cur_marker}.npz'))['data']
				saliency_img_path = os.path.join(f'{self.cumulative_images}/repetition_{cur_seed}', f'{self.method_name}_{sample_size}_alpha_{alpha_value}_marker_{cur_marker}_saliency')

				custom_plot_function(saliency_data, self.xlim, self.ylim, saliency_img_path, self.dpi_val, plot_type='image', color_map=plt.cm.gray,
										background_color=self.background, my_color=self.color,
											marker_size=self.marker, t=self.alpha, x_axis_size=self.x_canvas_size, y_axis_size=self.y_canvas_size)

	# Function to generate cumulative map for a sample          
	def overlay_intermediate_maps(self, sample_size, cur_seed):

		full_map = np.load(os.path.join(f'{self.cumulative_maps}/repetition_{cur_seed}', f'{self.method_name}_{sample_size}_alpha_{self.transparency_levels[0]}_marker_{self.marker_levels[0]}.npz'))['data']
		cumulative_saliency = np.zeros((full_map.shape[0], full_map.shape[1]))

		for cur_marker in self.marker_levels[self.marker_level_start:self.marker_level_end]:

			for alpha_value in self.transparency_levels[self.transparency_level_start:self.transparency_level_end]: 

				cur_saliency_map = np.load(os.path.join(f'{self.cumulative_maps}/repetition_{cur_seed}', f'{self.method_name}_{sample_size}_alpha_{alpha_value}_marker_{cur_marker}.npz'))['data']
				
				cur_saliency_map = max_min_normalization(cur_saliency_map)
				
				cumulative_saliency = np.maximum(cur_saliency_map, cumulative_saliency)

		# Save the cumulative saliency maps
		sal_path = os.path.join(f'{self.cumulative_maps}/repetition_{cur_seed}', f'{self.method_name}_{sample_size}_cumulative_saliency.npz')
		np.savez(sal_path, data=cumulative_saliency)

		saliency_img_path = os.path.join(f'{self.cumulative_images}/repetition_{cur_seed}',  f'{self.method_name}_{sample_size}_cumulative_saliency')

		# Plot the cumulative map
		custom_plot_function(cumulative_saliency, self.xlim, self.ylim, saliency_img_path, self.dpi_val, plot_type='image', color_map=plt.cm.gray,
								background_color=self.background, my_color=self.color,
									marker_size=self.marker, t=self.alpha, x_axis_size=self.x_canvas_size, y_axis_size=self.y_canvas_size)

	# Function to compute performance metrics. Saliency map is first read using an image format
	# Following prior work, we read the image and normalize it in the [0, 1] range 
	def compute_metrics(self, sample_size, marker_sizes, transparency_values, cur_seed, metric='ssim', x_resize=30, y_resize=30):

		scores = []
		point_sizes = []
		opacity_percents = []

		# Compute matrix for EMD function that assumes resized images
		if metric == 'emd':
			emd_cost_matrix = compute_distance_matrix(x_resize, y_resize)

		for i in range(len(marker_sizes)):

			for j in range(len(transparency_values)):

				reference_img = cv2.imread(os.path.join(self.data_saliency_path, f'cumulative_saliency/images/saliency_{self.dataset}_alpha_{transparency_values[j]}_marker_{marker_sizes[i]}.png')) 
				reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY) # this is a matrix from [0, 255]

				# Normalize in the [0, 1] range
				reference_img = max_min_normalization(reference_img)
				value_range = np.max(reference_img)-np.min(reference_img)

				# In case the image does not have any variation (e.g., a fully black or white image)
				if value_range == 0:
					value_range = 1 # default for structural similarity 

				sample_data_img = cv2.imread(os.path.join(f'{self.cumulative_images}/repetition_{cur_seed}', f'{self.method_name}_{sample_size}_alpha_{transparency_values[j]}_marker_{marker_sizes[i]}_saliency.png'))
				sample_data_img = cv2.cvtColor(sample_data_img, cv2.COLOR_BGR2GRAY) # this is a matrix from [0, 255]

				# Normalize in the [0, 1] range
				sample_data_img = max_min_normalization(sample_data_img)
				cur_score = -1

				if metric == 'ssim':
					# Compute SSIM on the image 
					cur_score, _ = structural_similarity(reference_img, sample_data_img, data_range=value_range, full=True)

				elif metric == 'js':
					cur_score = jensen_shannon_divergence(reference_img, sample_data_img)

				elif metric == 'pcc':
					cur_score = CC(reference_img, sample_data_img)

				elif metric == 'kl':
					cur_score = KL_divergence(reference_img, sample_data_img)

				elif metric == 'emd':
					# Resize the images
					reference_img_resized = cv2.resize(reference_img, (x_resize, y_resize))
					sample_data_img_resized = cv2.resize(sample_data_img, (x_resize, y_resize))

					cur_score = EMD(reference_img_resized, sample_data_img_resized, emd_cost_matrix)

				elif metric == 'sim':
					cur_score = SIM(reference_img, sample_data_img)

				scores.append(cur_score)
				point_sizes.append(marker_sizes[i])
				opacity_percents.append(transparency_values[j])

		
		statistics = {}
		statistics['raw_scores'] = scores
		statistics['point_sizes'] = point_sizes
		statistics['opacity_percents'] = opacity_percents

		statistics['average'] = np.mean(scores)
		statistics['max'] = np.max(scores)
		statistics['min'] = np.min(scores)
		statistics['argmax'] = np.argmax(scores)
		statistics['argmin'] = np.argmin(scores)
		statistics['std'] = np.var(scores)

		return statistics

	# Compare approximate saliency to PA saliency   
	def compute_metrics_for_approximate_technique(self, sample_size, marker_sizes, transparency_values, cur_seed, metric='ssim', x_resize=30, y_resize=30):

		scores = []
		point_sizes = []
		opacity_percents = []

		# Compute matrix for EMD function that assumes resized images
		if metric == 'emd':
			emd_cost_matrix = compute_distance_matrix(x_resize, y_resize)

		for i in range(len(marker_sizes)):

			for j in range(len(transparency_values)):

				reference_img = cv2.imread(os.path.join(f'{self.dataset}/sampling_techniques/perception_aware/cumulative_saliency/images/repetition_0', 
										  f'perception_aware_{sample_size}_alpha_{transparency_values[j]}_marker_{marker_sizes[i]}_saliency.png'))

				reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY) # this is a matrix from [0, 255]

				# Normalize in the [0, 1] range
				reference_img = max_min_normalization(reference_img)
				value_range = np.max(reference_img)-np.min(reference_img)

				# In case the image does not have any variation (e.g., a fully black or white image)
				if value_range == 0:
					value_range = 1 # default for structural similarity 

				sample_data_img = cv2.imread(os.path.join(f'{self.cumulative_images}/repetition_{cur_seed}', f'{self.method_name}_{sample_size}_alpha_{transparency_values[j]}_marker_{marker_sizes[i]}_saliency.png'))
				sample_data_img = cv2.cvtColor(sample_data_img, cv2.COLOR_BGR2GRAY) # this is a matrix from [0, 255]

				# Normalize in the [0, 1] range
				sample_data_img = max_min_normalization(sample_data_img)
				cur_score = -1

				if metric == 'ssim':
					# Compute SSIM on the image 
					cur_score, _ = structural_similarity(reference_img, sample_data_img, data_range=value_range, full=True)

				elif metric == 'js':
					cur_score = jensen_shannon_divergence(reference_img, sample_data_img)

				elif metric == 'pcc':
					cur_score = CC(reference_img, sample_data_img)

				elif metric == 'kl':
					cur_score = KL_divergence(reference_img, sample_data_img)

				elif metric == 'emd':
					# Resize the images
					reference_img_resized = cv2.resize(reference_img, (x_resize, y_resize))
					sample_data_img_resized = cv2.resize(sample_data_img, (x_resize, y_resize))

					cur_score = EMD(reference_img_resized, sample_data_img_resized, emd_cost_matrix)

				elif metric == 'sim':
					cur_score = SIM(reference_img, sample_data_img)

				scores.append(cur_score)
				point_sizes.append(marker_sizes[i])
				opacity_percents.append(transparency_values[j])

		
		statistics = {}
		statistics['raw_scores'] = scores
		statistics['point_sizes'] = point_sizes
		statistics['opacity_percents'] = opacity_percents

		statistics['average'] = np.mean(scores)
		statistics['max'] = np.max(scores)
		statistics['min'] = np.min(scores)
		statistics['argmax'] = np.argmax(scores)
		statistics['argmin'] = np.argmin(scores)
		statistics['std'] = np.var(scores)

		return statistics

# Baselines Methods 
class RandomSampler(Sampler):
	
	def __init__(self, dataset_name, experimental_settings):

		name = dataset_name
		method = 'random'

		super().__init__(name, method, experimental_settings)

	def compute_sample(self, sample_size, cur_seed, use_seed=True, save_sample=True):

		# Access the weighted dataset 
		path_to_weighted_data = os.path.join(self.parent_path, 'data/weighted.csv')

		data_df = pd.read_csv(path_to_weighted_data)
		data_np = data_df.to_numpy()

		N = data_np.shape[0]

		if use_seed:
			np.random.seed(cur_seed)
		
		start = time.time()
		
		sample_ids = np.random.choice(N, replace=False, size=sample_size)
		sample = data_np[sample_ids]

		end = time.time()

		elapsed_time = end - start

		if save_sample:
			self.store_sample(sample, sample_size, cur_seed)

		return sample_ids, data_np[sample_ids], elapsed_time

class MaxMinSampler(Sampler):

	def __init__(self, dataset_name, experimental_settings):

		name = dataset_name
		method = 'max_min'

		super().__init__(name, method, experimental_settings)

	def compute_sample(self, sample_size, init_points, cur_seed, use_seed=True, save_sample=True):

		# Access the weighted dataset 
		path_to_weighted_data = os.path.join(self.parent_path, 'data/weighted.csv')

		data_df = pd.read_csv(path_to_weighted_data)
		data_np = data_df.to_numpy()

		start = time.time()
		
		sample_indices = max_min_scalable_with_initial_points(data_np[:, 0], data_np[:, 1], init_points, cur_seed, use_seed, sample_size)
		sample = data_np[sample_indices]

		end = time.time()

		elapsed_time = end - start

		if save_sample:
			self.store_sample(sample, sample_size, cur_seed)

		return sample_indices, data_np[sample_indices], elapsed_time

class VisualizationAwareSampler(Sampler):

	def __init__(self, dataset_name, experimental_settings):

		name = dataset_name
		method = 'vas'

		super().__init__(name, method, experimental_settings)

	def estimateK(self, data, sample_ids, new_point, epsilon):
		distances_row = np.sqrt(np.sum(np.square(data[sample_ids] - data[new_point]), axis=1))
		return np.exp(-( distances_row**2 / (2 * epsilon**2) ))

	def expand(self, data, solution_ids, rsp_values, new_point, epsilon):
		k_values = self.estimateK(data, solution_ids, new_point, epsilon)

		solution_ids = np.append(solution_ids, new_point)

		rsp_values += k_values
		rsp_values = np.append(rsp_values, np.sum(k_values))
		
		return solution_ids, rsp_values     

	def shrink(self, data, solution_ids, rsp_values, epsilon):

		# Find the index of the max rsp value in rsp_values and the corresponding point in solution_ids. 
		max_index = np.argmax(rsp_values)
		point_to_remove = solution_ids[max_index]

		solution_ids = np.delete(solution_ids, max_index)

		rsp_values = np.delete(rsp_values, max_index)
		rsp_values -= self.estimateK(data, solution_ids, point_to_remove, epsilon)
		
		return solution_ids, rsp_values

	def compute_sample(self, sample_size, cur_seed, use_seed=True, save_sample=True):

		# Access the weighted dataset 
		path_to_weighted_data = os.path.join(self.parent_path, 'data/weighted.csv')

		data_df = pd.read_csv(path_to_weighted_data)
		data_np = data_df.to_numpy()

		# Only get the fist 2 columns
		data_np = data_np[:, :2]

		if use_seed:
			np.random.seed(cur_seed)

		start = time.time()
			
		indices = np.random.permutation(data_np.shape[0])
		epsilon = np.sqrt(data_np.shape[1])/100 

		solution_ids, rsp_values = indices[:sample_size], []

		for index in solution_ids:
			rsp_values.append(np.sum(self.estimateK(data_np, solution_ids, index, epsilon)))

		for index in indices[sample_size:]: 

			card = len(solution_ids)       
			solution_ids, rsp_values = self.expand(data_np, solution_ids, rsp_values, index, epsilon)

			if card >= sample_size:   
				solution_ids, rsp_values = self.shrink(data_np, solution_ids, rsp_values, epsilon)

		cur_end = time.time()

		one_pass_time = cur_end - start

		if not save_sample: # only monitoring time for one pass
			return solution_ids, data_np[solution_ids], one_pass_time

		# Code to improve VAS solution        
		num_iters, prev_total = 0, 0

		while True:

			indices = np.random.permutation(data_np.shape[0])

			for index in indices:
				if index not in solution_ids:
						solution_ids, rsp_values = self.expand(data_np, solution_ids, rsp_values, index, epsilon)
						solution_ids, rsp_values = self.shrink(data_np, solution_ids, rsp_values, epsilon)
						
			curr_total = np.sum(rsp_values)
			delta = np.square(curr_total - prev_total)
			prev_total = curr_total

			num_iters += 1

			# Condition 2: Check if solution stops improving
			if delta < 1e-4:
				created_sample = data_np[solution_ids]

				end = time.time()
				elapsed_time = end - start

				if save_sample:
					self.store_sample(data_np[solution_ids], sample_size, cur_seed)

				return solution_ids, data_np[solution_ids], elapsed_time

class DensityBiasedSampler(Sampler):

	def __init__(self, dataset_name, experimental_settings):

		name = dataset_name
		method = 'density_biased'

		super().__init__(name, method, experimental_settings)

	def compute_sample(self, sample_size, cur_seed, use_seed=True, save_sample=True):

		# Access the weighted dataset 
		path_to_weighted_data = os.path.join(self.parent_path, 'data/weighted.csv')

		data_df = pd.read_csv(path_to_weighted_data)
		data_np = data_df.to_numpy()

		# Only get the fist 2 columns
		data_np = data_np[:, :2]

		if use_seed:
			np.random.seed(cur_seed)

		start = time.time()

		k = 50

		N, d = data_np.shape

		if k + 1 > N:
			k = int((N - 1) / 2)

		knn = NearestNeighbors(n_neighbors=k+1)
		knn.fit(data_np)

		# Dist is sorted such that for every point the first neighbor is the closest
		dist, neighbor = knn.kneighbors(data_np, return_distance=True)

		# We take the neighbors that are farthest away for each point
		radius_of_k_neighbor = dist[:, -1]

		maxD = np.max(radius_of_k_neighbor)
		minD = np.min(radius_of_k_neighbor)

		if maxD - minD == 0:
			scale = 1
		else:
			scale = maxD - minD 

		radius_of_k_neighbor = ((radius_of_k_neighbor - minD) / scale) * 0.5 + 0.5

		prob = radius_of_k_neighbor
		prob = prob / prob.sum()
		
		selected_indexes = np.random.choice(N, sample_size, replace=False, p=prob)
		sample = data_np[selected_indexes]

		end = time.time()   

		elapsed_time = end - start

		if save_sample:
			self.store_sample(sample, sample_size, cur_seed)

		return selected_indexes, sample, elapsed_time

# Blue-noise Sampling
class BlueNoiseSampler(Sampler):

	def __init__(self, dataset_name, experimental_settings):

		name = dataset_name
		method = 'blue_noise'

		super().__init__(name, method, experimental_settings)

	def adjust_sample_size(self, N, m):
		if m < 0:
			raise ValueError('sample size must be positive')
		else:
			return int(m) if m >= 1 else round(N*m) 

	def reduce_num(self, num, factor=10, max_step=200, arr=[]):
		if num < max_step:
			arr.append(int(num)+1)
			return arr
		next_num = num/factor
		arr.append(factor+1)

		return self.reduce_num(next_num, factor, max_step, arr)

	def compute_sample(self, size, cur_seed, use_seed=True, save_sample=True, blue_noise_fail_rate = 0.1):

		# Access the weighted dataset 
		path_to_weighted_data = os.path.join(self.parent_path, 'data/weighted.csv')

		data_df = pd.read_csv(path_to_weighted_data)
		data_np = data_df.to_numpy()

		if use_seed:
			np.random.seed(cur_seed)

		n, d = data_np.shape

		m = size #round(n * self.sampling_rate)

		selected_indexes = []           
		sampling_rate = size / n
		
		start = time.time()

		k = int(1 / sampling_rate)
		knn = NearestNeighbors(n_neighbors=k+1)
		knn.fit(data_np)

		# Dist is sorted such that for every point the first neighbor is the closest
		dist, neighbor = knn.kneighbors(data_np, return_distance=True)
		radius = np.average(np.sqrt(dist[:, -1])) #np.average(np.sqrt(dist[:, 0])) #np.average(np.sqrt(dist[:, -1]))
		#print("R", radius)

		count = 0
		
		while count < m:
			failure_tolerance = min(5000, (n - m) * blue_noise_fail_rate)
			perm = np.random.permutation(n)
			fail = 0
			for idx in perm:
				if fail > failure_tolerance or count >= m:
					break
				
				success = True
				
				for selected_id in selected_indexes:
					if sum((data_np[idx] - data_np[selected_id])**2) < radius**2:
						success = False
						break
						
				if success:
					count += 1
					selected_indexes.append(idx)
				else:
					fail += 1
			radius /= 2
		
		end = time.time()
		elapsed_time = end-start

		selected_indexes = np.array(selected_indexes)

		if save_sample:     
			self.store_sample(data_np[selected_indexes], size, cur_seed)

		return selected_indexes, data_np[selected_indexes], elapsed_time

	#Modified code to make blue-noise scalable to large datasets, i.e, hidden correlation   
	def compute_sample_for_large_data(self, sample_size, cur_seed, use_seed=True, save_sample=True, use_step=True, max_step=25, return_radius=False, verbose=False):

		# Access the weighted dataset 
		path_to_weighted_data = os.path.join(self.parent_path, 'data/weighted.csv')

		data_df = pd.read_csv(path_to_weighted_data)
		data_np = data_df.to_numpy()

		if use_seed:
			np.random.seed(cur_seed)

		blue_noise_fail_rate = 0.1
		selected_indexes = []

		start = time.time()

		N, _ = data_np.shape

		sampling_rate = sample_size / N 

		total_k = int(1/sampling_rate) + 1

		if use_step and total_k != 1:
			ks = self.reduce_num(total_k, max_step=max_step, arr=[])
		else:
			ks = np.array([total_k]) + 1

		ks = np.sort(ks)[::-1]

		out = data_np

		for k_idx, k in enumerate(ks):

			n, _ = out.shape

			mp = m if not use_step or k_idx == len(ks)-1 else int(n/(k-1))
			
			failure_tolerance = min(5000, (n - mp) * blue_noise_fail_rate)

			if verbose:
				print('current data size:',n)
				print('k:', k)

			knn = NearestNeighbors(n_neighbors=k)   
			knn.fit(out)
			dist, _ = knn.kneighbors(out, return_distance=True)
			radius = np.average(dist[:, -1])
			
			if verbose:
				print('computing k neighbors and radius: t =', time.time()-start)
			
			count, unseen, subset, out_data = 0, None, [], []        

			start = time.time()

			while count < mp:
				this_round = time.time()

				if verbose:
					print('radius:',radius)
				
				fail, unseen = 0, []
				perm = np.random.permutation(unseen or n)

				for i, idx in enumerate(perm):
					if fail > failure_tolerance or count >= mp:
						unseen += perm[i:].tolist()
						break
					if len(subset) == 0:
						success = True
					else:
						success = np.min(np.sum(np.square(np.array(subset)-out[idx]), axis=1)) >= radius**2

					if success:
						count += 1
						subset.append(out[idx])
						out_data.append(out[idx])
						selected_indexes.append(idx)
					else:
						unseen.append(idx)
						fail += 1

				radius /= 2

				if verbose:
					print(f'current count {count}, t = {time.time()-this_round}')

			if verbose:
				print('picking points: t =', time.time()-start)
				print()

			out = out_data 

			k /= 10

		end = time.time()
		elapsed_time = end - start  

		if save_sample:     
			self.store_sample(out, sample_size, cur_seed)

		if return_radius:
			return out, radius*2
		
		return selected_indexes, out, elapsed_time

# Perception-aware Sampling Methods
class PerceptionAwareSampler(Sampler):

	def __init__(self, dataset_name, experimental_settings):

		name = dataset_name
		method = 'perception_aware'

		super().__init__(name, method, experimental_settings)

	def compute_sample(self, sample_size, init_points, cur_seed, use_seed=True, save_sample=True):

		# Access the weighted dataset 
		path_to_weighted_data = os.path.join(self.parent_path, 'data/weighted.csv')

		data_df = pd.read_csv(path_to_weighted_data)
		data_np = data_df.to_numpy()

		start = time.time()

		sample_indices = max_min_scalable_with_initial_points_with_saliency(data_np[:, 0], data_np[:, 1], data_np[:, 2], init_points, cur_seed, use_seed, sample_size)
		sample = data_np[sample_indices]

		end = time.time()

		elapsed_time = end - start

		if save_sample:     
			self.store_sample(sample, sample_size, cur_seed)

		return sample_indices, data_np[sample_indices], elapsed_time

class PerceptionAwarewithDensitySampler(Sampler):

	def __init__(self, dataset_name, experimental_settings):

		name = dataset_name
		method = 'perception_aware_with_density'

		super().__init__(name, method, experimental_settings)

	# Code when sample size is not taken into account   
	def compute_sample(self, sample_size, init_points, cur_seed, use_seed=True, save_sample=True):

		# Access the weighted dataset with both saliency & density 
		path_to_weighted_data = os.path.join(self.parent_path, 'data/saliency_and_density_weights.csv')

		print("1", path_to_weighted_data)

		data_df = pd.read_csv(path_to_weighted_data)
		data_np = data_df.to_numpy()

		# Combine density & saliency weights using max operator
		combined_weights = np.maximum(data_np[:, 2], self.gamma_param*data_np[:, 3])

		plt.figure(figsize=(14, 12), facecolor='white')
		plt.scatter(data_np[:, 0], data_np[:, 1], c=combined_weights, s=2, alpha=1)

		# Set limits for the axes
		plt.xlim([-0.05, 1.05])
		plt.ylim([-0.05, 1.05])
		plt.show()

		start = time.time()

		sample_indices = max_min_scalable_with_initial_points_with_saliency(data_np[:, 0], data_np[:, 1], combined_weights, init_points, cur_seed, use_seed, sample_size)
		sample = data_np[sample_indices]

		end = time.time()

		elapsed_time = end - start
		#print("Total time", elapsed_time, "knn time", end-start_knn)

		if save_sample:     
			#path_to_raw_data = os.path.join(self.parent_path, 'data/weighted.csv')
			self.store_sample(sample, sample_size, cur_seed)

		return sample_indices, data_np[sample_indices], elapsed_time

class BoxSampling(Sampler):

	def __init__(self, dataset_name, experimental_settings, method='box_sampling'):

		name = dataset_name
		super().__init__(name, method, experimental_settings)

	def define_box(self, data, x_start, x_end, y_start, y_end):

		filtered_indices = np.where(data[:, 0] >= x_start)[0]
		subset_indices = np.where(data[filtered_indices, 0] <= x_end)[0]

		# Contains all the indices for the items filtered on the x-axis
		subspace_indices = filtered_indices[subset_indices]

		y_indices =  np.where(data[subspace_indices, 1] >= y_start)[0]

		subspace_indices = subspace_indices[y_indices]
		y_indices = np.where(data[subspace_indices, 1] <= y_end)[0]

		subspace_indices = subspace_indices[y_indices]

		return subspace_indices 

	# Function to apply the first-layer grid onto the data      
	def retrieve_grid_buckets(self, data):
		
		"""
			Inputs:
				-data: A numpy array of size (N, 2) that contains all points.
			Output:
				-data_buckets: A list of grid buckets, which are numpy arrays.
		"""
		data_buckets = []
		bucket_sizes = []

		num_bins = 4

		H, xedges, yedges = np.histogram2d(data[:, 0], data[:, 1], bins=num_bins) #bins=[x_bins, y_bins])

		# How to map points to bins: stackoverflow post
		# https://stackoverflow.com/questions/35961944/retrieving-bin-data-from-2d-histograms-in-numpy
		xedges[-1] += 1e-6 # add a small epsilon value to guarantee that the point with xedges[-1] falls into the last bucnet
		yedges[-1] += 1e-6

		x_assignment = np.digitize(data[:, 0], xedges)
		y_assignment = np.digitize(data[:, 1], yedges)

		final_assignment = list(zip(y_assignment, x_assignment))
		final_assignment_np = np.asarray(final_assignment)

		# Unique bucket combinations
		unique_values = np.unique(np.asarray(final_assignment), axis=0)
		
		for i in range(unique_values.shape[0]):
			
			# For every row of the assignment check which points fall in that bucket
			ind = (final_assignment_np == unique_values[i]).all(axis=1)

			data_buckets.append(data[ind])
			bucket_sizes.append(data[ind].shape[0])

		# Check if you managed to assign all points inside a buckets
		if np.sum(bucket_sizes) < data.shape[0]:
			assert True, "Failed to assign all points into a grid buckets"
			
		return data_buckets, bucket_sizes, num_bins

	""" 
	Function where we use an "approximation" based criterion to keep splitting the boxes. How well the box represents the original distribution.
	"""
	def compute_average_CD_distance(self, sample, original_data):

		# Distance of the sample to the original data
		distance_set_sample = euclidean_distances(sample, original_data)
		min_distances_sample = np.min(distance_set_sample, axis=1)

		# Distance of the original data to the sample
		distance_set_data = euclidean_distances(original_data, sample)
		min_distances_data = np.min(distance_set_data, axis=1)

		chamfer_distance = np.mean(min_distances_sample) + np.mean(min_distances_data)

		# Average the score
		chamfer_distance /= 2

		return chamfer_distance

	def recursive_split_boxes_with_saliency(self, my_array, approximation_threshold, sal_threshold,
												x_start, x_end, y_start, y_end, number_of_points=500, cur_seed=0, use_seed=True):

		S = time.time()

		if use_seed:
			np.random.seed(cur_seed)

		my_list = [] # stores the splits
		my_queue = []
		x_queue = []
		y_queue = []

		my_queue.append(my_array)
		x_queue.append([x_start, x_end])
		y_queue.append([y_start, y_end])

		# initialize this with infinity 
		split_distance = np.inf

		while my_queue:

			cur_subspace = my_queue.pop(0) # parent subspace

			# Compute the number of points inside the current subspace
			num_points = cur_subspace.shape[0]  

			#saliency_variance = np.var(cur_subspace[:, 2]) 
			# Before splitting was happening only based on saliency weights -- now it should happen based on this combined weight
			perception_weights = np.maximum(cur_subspace[:, 2], self.gamma_param * cur_subspace[:, 3])
			saliency_variance = np.var(perception_weights) 

			# Pop limits
			x_start, x_end = x_queue.pop(0)
			y_start, y_end = y_queue.pop(0)

			x_middle = (x_start + x_end) / 2
			y_middle = (y_start + y_end) / 2

			# Compute an approximate sample
			xy_min = [x_start, y_start + 1e-8]
			xy_max = [x_end, y_end + 1e-8]

			sample_size = number_of_points 

			if sample_size > 0:
				approx_data = np.random.uniform(low=xy_min, high=xy_max, size=(sample_size,2))
				split_distance = self.compute_average_CD_distance(approx_data, cur_subspace[:, :2])

			else:
				split_distance = 0      

			if (split_distance <= approximation_threshold and saliency_variance <= sal_threshold) or cur_subspace.shape[0] == 1:
				my_list.append(cur_subspace)

			else: # keep spliting

				xedges = np.array([x_start, x_middle, x_end])
				yedges = np.array([y_start, y_middle, y_end])
				
				initial_indices = np.arange(cur_subspace.shape[0])

				# Object 1 - Bottom Left
				bottom_left_indices = self.define_box(cur_subspace, x_start, x_middle, y_start, y_middle)
				bottom_left = cur_subspace[bottom_left_indices]

				if bottom_left.shape[0] > 0:
					x_queue.append([x_start, x_middle])
					y_queue.append([y_start, y_middle])
					my_queue.append(bottom_left)

				remaining_indices = np.setdiff1d(initial_indices, bottom_left_indices)
				remaining_subspace = cur_subspace[remaining_indices]

				# Object 2 - Bottom Right               
				bottom_right_indices = self.define_box(remaining_subspace, x_middle, x_end, y_start, y_middle)
				bottom_right = remaining_subspace[bottom_right_indices]
					
				if bottom_right.shape[0] > 0:
					x_queue.append([x_middle, x_end])
					y_queue.append([y_start, y_middle])
					my_queue.append(bottom_right)

				remaining_indices = np.setdiff1d(np.arange(remaining_subspace.shape[0]), bottom_right_indices)
				remaining_subspace = remaining_subspace[remaining_indices]

				# Object 3 - Upper left 
				upper_left_indices = self.define_box(remaining_subspace, x_start, x_middle, y_middle, y_end)
				upper_left = remaining_subspace[upper_left_indices]
				
				if upper_left.shape[0] > 0:
					x_queue.append([x_start, x_middle])
					y_queue.append([y_middle, y_end])
					my_queue.append(upper_left)

				# Update on 4/25: Remove all the points in bottom left from cur_subspace    
				remaining_indices = np.setdiff1d(np.arange(remaining_subspace.shape[0]), upper_left_indices)
				remaining_subspace = remaining_subspace[remaining_indices]  
				
				# Object 4 - Upper right
				upper_right_indices = self.define_box(remaining_subspace, x_middle, x_end, y_middle, y_end)
				upper_right = remaining_subspace[upper_right_indices]
				
				if upper_right.shape[0] > 0:
					x_queue.append([x_middle, x_end])
					y_queue.append([y_middle, y_end])
					my_queue.append(upper_right)

				assert upper_right.shape[0] == upper_right_indices.shape[0], "Something went wrong in splitting points"

		return my_list 

	# Main function to compute the partitions   
	def retrieve_boxes_with_saliency(self, approximation_threshold=1e-3, sal_threshold=1e-2, points=500):

		# Read the weighted data
		path_to_weighted_data = os.path.join(self.parent_path, 'data/saliency_and_density_weights.csv') #'data/weighted.csv')
		data_df = pd.read_csv(path_to_weighted_data)
		data_np = data_df.to_numpy()

		# Create buckets of the grid
		buckets, sizes, bin_number = self.retrieve_grid_buckets(data_np)

		final_splits = {}
		statistics = {} # every entry for every box you get

		for i in range(len(buckets)):

			cur_group = buckets[i]

			x_start = np.min(cur_group[:, 0]) 
			x_end = np.max(cur_group[:, 0])

			y_start = np.min(cur_group[:, 1])
			y_end = np.max(cur_group[:, 1])

			subsplits = self.recursive_split_boxes_with_saliency(cur_group, approximation_threshold, sal_threshold, 
																	x_start, x_end, y_start, y_end, points)
			final_splits[i] = subsplits
			statistics[i] = []

			for subsplit in subsplits:
				x_min = np.min(subsplit[:, 0])
				x_max = np.max(subsplit[:, 0]) 
				y_min = np.min(subsplit[:, 1])
				y_max = np.max(subsplit[:, 1])
				cur_stat = np.array([x_min, x_max, y_min, y_max])
				statistics[i].append(cur_stat)

		return final_splits, statistics

	def aggregate_boxes_information(self, saliency_group_boxes, saliency_group_statistics, cur_seed=0):

		# Serialize all the information in a single list
		boxes_info = []
		statistics_info = []
		saliency_info = []
		mean_saliency = []
		variance_saliency =[]
		boxes_sizes = []

		for key, value in saliency_group_boxes.items():
			#print(key, value)

			for j in range(len(value)):

				# var has the actual data
				var = value[j] # current sub-box
				boxes_info.append(var)

				# Info about the corner of the boxes
				statistics_info.append(saliency_group_statistics[int(key)][j])

				# Actual saliency weights of points
				#saliency_info.append(var[:, 2])
				#mean_saliency.append(np.mean(var[:, 2]))

				perception_weights = np.maximum(var[:, 2], self.gamma_param*var[:, 3])
				saliency_info.append(perception_weights)
				mean_saliency.append(np.mean(perception_weights))

				#variance_saliency.append(np.var(var[:, 2]))
				variance_saliency.append(np.var(perception_weights))
				boxes_sizes.append(var.shape[0])
		"""
		# Store the serialized information about the approximate sampling technique into an npz file
		np.savez(f'{self.cumulative_data}/approximate_scheme_with_data_per_box.npz', 
						data_per_box=np.asarray(boxes_info, dtype='object'), coordinates_per_box=np.asarray(statistics_info, dtype='object'), 
							data_weights_per_box=np.asarray(saliency_info, dtype='object'), average_saliency_per_box=np.asarray(mean_saliency), 
								saliency_variance_per_box=np.asarray(variance_saliency), sizes_per_box=np.asarray(boxes_sizes))
	
		"""
		np.savez(f'{self.cumulative_data}/approximate_scheme.npz', 
						coordinates_per_box=np.asarray(statistics_info, dtype='object'), 
							average_saliency_per_box=np.asarray(mean_saliency), 
								sizes_per_box=np.asarray(boxes_sizes))

		return boxes_info, statistics_info, saliency_info, mean_saliency, boxes_sizes

	def map_to_box_coordinates(self, draws, boxes_info, boxes_saliency, num_points_per_box):

		data_x = []
		data_y = []
		
		weights = []
		labels = []

		cumulative = np.cumsum(num_points_per_box) 
		start_points = np.concatenate(([0], cumulative[:-1])) 
		end_points = cumulative

		for j in range(len(boxes_info)):

			coordinates = boxes_info[j]

			x_min, x_max, y_min, y_max = coordinates[0], coordinates[1], coordinates[2], coordinates[3]
			
			# Change to account for the fact that boxes do not have the same of representatives sampled
			start_point = start_points[j] #j*num_points_per_box
			end_point = end_points[j] #start_point + num_points_per_box
			
			draws_for_current_box = draws[start_point:end_point]

			x_coords = (x_max - x_min)*draws_for_current_box[:, 0] + x_min
			y_coords = (y_max-y_min)*draws_for_current_box[:, 1] + y_min
			
			data_x.extend(x_coords)
			data_y.extend(y_coords)
			
			weights.extend([boxes_saliency[j]]*num_points_per_box[j])
			labels.extend([j]*num_points_per_box[j])

		data = np.ones((len(data_x), 2))
		data[:, 0] = data_x
		data[:, 1] = data_y

		return data, np.asarray(weights), np.asarray(labels)    

	def init_representatives(self, boxes_sizes, boxes_info, boxes_saliency, num_reps_per_box, cur_seed, use_seed=True):

		if use_seed:
			np.random.seed(cur_seed)

		total_num_of_points = np.sum(num_reps_per_box)

		low_bound = 0
		upper_bound = 1 + 1e-8

		random_draws = np.random.uniform(low=low_bound, high=upper_bound, size=(total_num_of_points, 2))
		data_np, saliency_weights, box_labels = self.map_to_box_coordinates(random_draws, boxes_info, boxes_saliency, num_reps_per_box)

		return data_np, saliency_weights, box_labels

	def draw_point_of_box(self, box_info, draw_info):
			
		x_min, x_max, y_min, y_max = box_info[0], box_info[1], box_info[2], box_info[3]
		
		point_x_coord = (x_max - x_min)* draw_info[0] + x_min
		point_y_coord = (y_max-y_min)* draw_info[1] + y_min
		
		return point_x_coord, point_y_coord 

	def approx_pa(self, sampling_budget, sizes, coordinates, saliency, num_points, cur_seed, use_seed=True):

		if use_seed:
			np.random.seed(cur_seed)

		sample_x = []
		sample_y = []

		# array to keep track how many points we picked from a box
		points_picked = np.zeros(len(sizes)).astype('int')

		"""
		print("Inside approx_pa")

		fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
		color_map= plt.cm.viridis
		# Plot the boxes
		for cc in range(len(coordinates)):

			splits = coordinates[cc]

			# Corner Orders: BL, BR, UL, UR
			rectangle = [(splits[0], splits[2]), (splits[1], splits[2]), (splits[0], splits[3]), (splits[1], splits[3])]

			xy = rectangle[0] 
			width = (splits[1] - splits[0]) #*scale_factors[count]  # xmax - xmin, rectangle[3][0] - rectangle[0][0]
			height = (splits[3] - splits[2]) #*scale_factors[count] # ymax-ymin, rectangle[3][1] - rectangle[0][1]

			rect = Rectangle(xy, width, height, linewidth=5, edgecolor='none', facecolor=color_map(saliency[cc]), alpha=1)

			axs.add_patch(rect)

			# Axis 
			axs.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
			axs.set_xticks([])
			axs.set_yticks([])

		plt.show()
		"""

		# Step 1: Materialize a number of items per box
		data, weights, labels = self.init_representatives(sizes, coordinates, saliency, num_points, cur_seed, use_seed)
		
		# Info to know which points cannot be picked --- once a box has reached full capacity
		cumulative = np.cumsum(num_points) 
		start_indices = np.concatenate(([0], cumulative[:-1])) 
		end_indices = cumulative

		#print(data.shape, weights.shape)
		#plt.figure(figsize=(14, 12))
		#plt.scatter(data[:, 0], data[:, 1], c=weights, s=1, alpha=1) 
		#plt.show()

		# Step 2: Draw k random points when it is time to replace a representative
		low_bound = 0
		upper_bound = 1 + 1e-8
		draw_replacements = np.random.uniform(low=low_bound, high=upper_bound, size=(sampling_budget, 2))
		
		# Step 3: Select the first point
		first_index = np.random.randint(len(data))
		point = data[first_index].copy()
		
		sample_x.append(point[0])
		sample_y.append(point[1])
		
		box_ID = labels[first_index]
		points_picked[box_ID] += 1
		
		# Replace the point with a new point    
		new_point_x, new_point_y = self.draw_point_of_box(coordinates[box_ID], draw_replacements[0])
		data[first_index] = [new_point_x, new_point_y]

		# Compute distances 
		euclidean_distances = dis_matrix(data[:, 0], data[:, 1], point[0], point[1])
		
		# Check if box reached capacity
		if points_picked[box_ID] == sizes[box_ID]:

			start_index = start_indices[box_ID]#box_ID * num_points
			end_index = end_indices[box_ID] #start_index + num_points
			euclidean_distances[start_index:end_index] = -np.inf
			
		distances = euclidean_distances * weights
		
		it = 1

		while (len(sample_x) < sampling_budget):

			next_id = np.argmax(distances)
			box_ID = labels[next_id]
			
			# Point to be appended into the sample
			sampled_point = data[next_id].copy()
			points_picked[box_ID] += 1
			
			# Replace the point picked 
			new_point_x, new_point_y = self.draw_point_of_box(coordinates[box_ID], draw_replacements[it])

			# Compute distances of the new point to the current sample
			sample_distances = dis_matrix(np.asarray(sample_x), np.asarray(sample_y), new_point_x, new_point_y)
			cur_min = np.min(sample_distances)

			euclidean_distances[next_id] = cur_min

			#if len(sample_x) > 4000:
			#   plt.figure(figsize=(14, 12))
			#   plt.scatter(data[:, 0], data[:, 1], c=weights, s=8)
			#   plt.scatter(data[next_id][0], data[next_id][1], s=30, c=weights[next_id])
			#   plt.scatter(new_point_x, new_point_y, s=20, c='red')
			#   plt.show()

			# Replace the data with the new point 
			data[next_id] = [new_point_x, new_point_y]

			# Find the distances wrt to the newly added point into the sample
			intermediate_dis_mat = dis_matrix(data[:, 0], data[:, 1], sampled_point[0], sampled_point[1])           
			euclidean_distances = np.minimum(euclidean_distances, intermediate_dis_mat)

			# Check if box reached capacity
			if points_picked[box_ID] == sizes[box_ID]:
				start_index = start_indices[box_ID] #box_ID * num_points
				end_index = end_indices[box_ID] #start_index + num_points
				euclidean_distances[start_index:end_index] = -np.inf
				
			distances = euclidean_distances * weights
			
			# Append the new point to the sample
			sample_x.append(sampled_point[0])
			sample_y.append(sampled_point[1])

			it += 1
		
		assert len(sample_x) == sampling_budget, "Did not satisfy sampling budget in approximate max-min"
		assert (sum(points_picked <= sizes)) == len(sizes), "Picked more points from a group than the group actually had"

		return sample_x, sample_y, points_picked 

	def compute_sample(self, sample_size, cur_seed, use_seed=True, save_sample=True):

		my_sample = []

		# Read the information about the approximation scheme
		filepath = os.path.join(os.path.join(self.cumulative_data, f'approximate_scheme.npz'))
		info = np.load(filepath, allow_pickle=True)

		coordinates_per_box = info['coordinates_per_box'] 
		average_saliency_per_box = info['average_saliency_per_box']
		sizes_per_box = info['sizes_per_box']

		# Compute area per box -- this is needed to distribute the points proportionally to area size
		area_per_box = []

		for cc in range(len(coordinates_per_box)):
			x_len = coordinates_per_box[cc, 1] - coordinates_per_box[cc, 0]
			y_len = coordinates_per_box[cc, 3] - coordinates_per_box[cc, 2]

			area = x_len * y_len
			area_per_box.append(area)

		print("Number of boxes", len(coordinates_per_box))
		"""
		print(len(sizes_per_box), "plot the scheme")
		
		fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
		color_map= plt.cm.viridis
		# Plot the boxes
		for cc in range(len(coordinates_per_box)):

			splits = coordinates_per_box[cc]

			# Corner Orders: BL, BR, UL, UR
			rectangle = [(splits[0], splits[2]), (splits[1], splits[2]), (splits[0], splits[3]), (splits[1], splits[3])]

			xy = rectangle[0] 
			width = (splits[1] - splits[0]) #*scale_factors[count]  # xmax - xmin, rectangle[3][0] - rectangle[0][0]
			height = (splits[3] - splits[2]) #*scale_factors[count] # ymax-ymin, rectangle[3][1] - rectangle[0][1]

			rect = Rectangle(xy, width, height, linewidth=5, edgecolor='none', facecolor=color_map(average_saliency_per_box[cc]), alpha=1)

			axs.add_patch(rect)

			# Axis 
			axs.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
			axs.set_xticks([])
			axs.set_yticks([])

			#if j == 0:
			#   axs[i].set_ylabel(rf'$\lambda={round(approx_threshold, 3)}$', fontsize=20)

			#if i == 0:
			#       axs[i].set_title(rf'$\sigma={round(saliency_threshold, 4)}$', fontsize=20)

		plt.show()
		"""

		C_reps = number_of_reps[self.dataset][self.method_name]

		total_budget = C_reps *len(coordinates_per_box)
		percentages = area_per_box / np.sum(area_per_box)

		# Distribute the budget proportionally 
		number_reps_per_box = np.maximum(np.minimum(1, sizes_per_box), np.floor(percentages * total_budget).astype('int'))
		#total_num_of_points = np.sum(final_num_points)

		start = time.time()
		x_sample, y_sample, _ = self.approx_pa(sample_size, sizes_per_box, coordinates_per_box, average_saliency_per_box, number_reps_per_box, cur_seed, use_seed)

		my_sample = np.hstack((np.asarray(x_sample).reshape(sample_size, 1), np.asarray(y_sample).reshape(sample_size, 1)))

		end = time.time()

		elapsed_time = end-start

		if save_sample:
			self.store_sample(my_sample, sample_size, cur_seed)

		return my_sample, elapsed_time