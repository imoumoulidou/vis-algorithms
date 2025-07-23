"""
	This file contains all the methods needed to:
		-- produce an aggregate saliency map
		-- assign weights to a dataset 
"""

import numpy as np
import pandas as pd
import os
import matlab.engine
import json
import time
import cv2

from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KernelDensity
from scipy.ndimage import gaussian_filter  
from utils import *

class Dataset():

	def __init__(self, data_path, saliency_path, sampling_path, 
					x_attr, y_attr, size, name, epsilon, plotting_info):

		# Storage paths 
		self.data_path = data_path
		self.saliency_path = saliency_path
		self.sampling_path = sampling_path

		self.x_attr = x_attr 
		self.y_attr = y_attr
		self.size = size
		self.name = name

		# Plotting information
		self.x_canvas_size = plotting_info['x_canvas_size']
		self.y_canvas_size = plotting_info['y_canvas_size']
		self.dpi_val = plotting_info['dpi_value']
		self.marker = plotting_info['marker']
		self.alpha = plotting_info['alpha']
		self.xlim = plotting_info['x_lim']
		self.ylim = plotting_info['y_lim']
		self.color = plotting_info['color']
		self.background = plotting_info['background_color']

		# Resize images to common canvas size because removing padding results to different sizes 
		self.resize_x = plotting_info['resize_x']
		self.resize_y = plotting_info['resize_y']

		# Attributes for coming up with cumulative saliency
		self.transparency_levels = plotting_info['transparency_values'] 
		self.marker_levels = plotting_info['marker_values']

		# Parameters for generating aggregate maps
		self.marker_level_start = plotting_info['marker_level_start']
		self.marker_level_end = plotting_info['marker_level_end']

		self.transparency_level_start = plotting_info['transparency_level_start']
		self.transparency_level_end = plotting_info['transparency_level_end']

		self.cm_saliency_img_path = os.path.join(self.saliency_path, 'cumulative_saliency/images')
		self.cm_saliency_map_path = os.path.join(self.saliency_path, 'cumulative_saliency/maps')

		# Info for approximate partitioning
		self.epsilon = epsilon

		# Info for kernel density estimation 
		self.kernel = plotting_info['kernel']
		self.kernel_bandwidth = plotting_info['kernel_bandwidth']

	def create_storage_paths(self):

		# Create folder to store raw and processed data
		if not os.path.exists(self.data_path):
			os.makedirs(self.data_path)

		# Create folder to store saliency data
		if not os.path.exists(self.saliency_path):
			os.makedirs(self.saliency_path)

		# Create folder to store sampling info 
		if not os.path.exists(self.sampling_path):
			os.makedirs(self.sampling_path)

	# Max-Min normalization across features for a dataset   
	def normalize_data(self):

		# Read the raw dataset into a pandas dataframe
		path_to_raw_data = os.path.join(self.data_path, 'raw.csv')
		data_df  = pd.read_csv(path_to_raw_data)

		# Select the attributes we are interested in
		cur_data = data_df[[self.x_attr, self.y_attr]]
		data_np = cur_data.to_numpy()

		# Normalize and store into cvs file
		data_storage_path = os.path.join(self.data_path, 'normalized.csv')
		normalized_data = MinMaxScaler().fit_transform(data_np)

		self.size = normalized_data.shape[0]

		df = pd.DataFrame(normalized_data, columns=['X', 'Y'])
		df.to_csv(data_storage_path, index=False)

	# Function to plot the cumulative saliency for a dataset    
	def plot_cumulative_saliency(self, colormap=plt.cm.viridis):

		# Read saliency data 
		saliency_data = np.load(os.path.join(self.saliency_path, 'cumulative_full_saliency.npz'))['data']
		saliency_img_path = os.path.join(self.saliency_path, f'cumulative_saliency_map')

		custom_plot_function(saliency_data, self.xlim, self.ylim, saliency_img_path, self.dpi_val, plot_type='image', color_map=colormap,
								background_color=self.background, my_color=self.color,
									marker_size=self.marker, t=self.alpha, x_axis_size=self.x_canvas_size, y_axis_size=self.y_canvas_size)

	# Function to plot the cropped cumulative saliency for a dataset    
	def plot_cropped_cumulative_saliency(self, colormap=plt.cm.viridis):

		# Read saliency data 
		saliency_data = np.load(os.path.join(self.saliency_path, 'cropped_cumulative_full_saliency.npz'))['data']
		saliency_img_path = os.path.join(self.saliency_path, f'cropped_cumulative_saliency_map')

		custom_plot_function(saliency_data, self.xlim, self.ylim, saliency_img_path, self.dpi_val, plot_type='image', color_map=colormap,
								background_color=self.background, my_color=self.color,
									marker_size=self.marker, t=self.alpha, x_axis_size=self.x_canvas_size, y_axis_size=self.y_canvas_size)

	# Code to generate saliency that takes into account multiple level of opacity and point size                                    
	def generate_intermediate_saliency_maps(self):

		if not os.path.exists(self.cm_saliency_img_path):
			os.makedirs(self.cm_saliency_img_path)

		if not os.path.exists(self.cm_saliency_map_path):
			os.makedirs(self.cm_saliency_map_path)  

		# Read normalized data
		data_df = pd.read_csv(os.path.join(self.data_path, 'normalized.csv'))
		data_np = data_df.to_numpy()

		eng = matlab.engine.start_matlab()

		# Phase 1: Generate all possible stimuli 
		for cur_marker in self.marker_levels:

			for alpha_value in self.transparency_levels: 

				img_path = f'{self.cm_saliency_img_path}/{self.name}_alpha_{alpha_value}_marker_{cur_marker}'
				
				######### Image generation ########
				custom_plot_function(data_np, self.xlim, self.ylim, img_path, self.dpi_val, 
										background_color=self.background, my_color=self.color,
											marker_size=cur_marker, t=alpha_value, x_axis_size=self.x_canvas_size, y_axis_size=self.y_canvas_size) 
				######################################

		# Phase 2: Generate all the possible saliency maps & their cropped version.

		""" To assign weights to the points, we need to have a cropped saliency. We define the cropped area
			using the image of the dataset with the higher marker size + transparency since this is the setting
			where the dataset spans higher volume onto the canvas. Assume that the highest transparency + marker level
			are the last entries.
		""" 

		img_name = f'{self.cm_saliency_img_path}/{self.name}_alpha_{self.transparency_levels[-1]}_marker_{self.marker_levels[-1]}.png'
		cropped_columns, cropped_rows = eng.crop_image(img_name, nargout=2)

		data_columns = np.asarray(cropped_columns[0]).astype('int')
		data_columns = data_columns.reshape((data_columns.shape[1],))
		data_rows = np.asarray(cropped_rows[0]).astype('int')
		data_rows = data_rows.reshape((data_rows.shape[1],))

		for cur_marker in self.marker_levels:

			for alpha_value in self.transparency_levels:

				img_path = f'{self.cm_saliency_img_path}/{self.name}_alpha_{alpha_value}_marker_{cur_marker}'

				######## Saliency generation ########
				full_saliency_path = os.path.join(self.cm_saliency_map_path, f'full_saliency_alpha_{alpha_value}_marker_{cur_marker}.npz')
				
				full_saliency = eng.generate_full_saliency(f'{img_path}.png')
				full_saliency = np.array(full_saliency)
				np.savez(full_saliency_path, data=full_saliency)

				full_saliency = np.load(full_saliency_path)['data']
				plt.imshow(full_saliency)

				# Plot the saliency map #
				saliency_img_path = f'{self.cm_saliency_img_path}/saliency_{self.name}_alpha_{alpha_value}_marker_{cur_marker}'

				custom_plot_function(full_saliency, self.xlim, self.ylim, saliency_img_path, self.dpi_val, plot_type='image', color_map=plt.cm.gray,
										background_color=self.background, my_color=self.color,
											marker_size=cur_marker, t=alpha_value, x_axis_size=self.x_canvas_size, y_axis_size=self.y_canvas_size) 

				# Weighted Scheme: Derive cropped saliency
				cropped_saliency = full_saliency[data_rows[0]:data_rows[-1]+1, data_columns[0]:data_columns[-1]+1]
				cropped_saliency_path = os.path.join(self.cm_saliency_map_path, f'cropped_saliency_alpha_{alpha_value}_marker_{cur_marker}.npz')
				plt.imshow(cropped_saliency)

				np.savez(cropped_saliency_path, data=cropped_saliency)
				######################################

		eng.quit()

	# Overlay all the generated to generate the aggregate saliency map  
	def overlay_saliency_maps(self):

		print('Overlaying heatmaps')

		full_map = np.load(os.path.join(self.cm_saliency_map_path, f'full_saliency_alpha_{self.transparency_levels[0]}_marker_{self.marker_levels[0]}.npz'))['data']
		cumulative_saliency = np.zeros((full_map.shape[0], full_map.shape[1]))

		# Read the first cropped_map to get access to the dimensions
		crop_map = np.load(os.path.join(self.cm_saliency_map_path, f'cropped_saliency_alpha_{self.transparency_levels[0]}_marker_{self.marker_levels[0]}.npz'))['data']
		cropped_cumulative_saliency = np.zeros((crop_map.shape[0], crop_map.shape[1]))

		for cur_marker in self.marker_levels[self.marker_level_start:self.marker_level_end]:

			for alpha_value in self.transparency_levels[self.transparency_level_start:self.transparency_level_end]:

				cur_saliency_map = np.load(os.path.join(self.cm_saliency_map_path, f'full_saliency_alpha_{alpha_value}_marker_{cur_marker}.npz'))['data']
				cur_cropped_saliency_map = np.load(os.path.join(self.cm_saliency_map_path, f'cropped_saliency_alpha_{alpha_value}_marker_{cur_marker}.npz'))['data']

				cur_saliency_map = max_min_normalization(cur_saliency_map)
				cur_cropped_saliency_map = max_min_normalization(cur_cropped_saliency_map)

				cumulative_saliency = np.maximum(cur_saliency_map, cumulative_saliency)
				cropped_cumulative_saliency = np.maximum(cur_cropped_saliency_map, cropped_cumulative_saliency)

				plt.imshow(cropped_cumulative_saliency)
				plt.show()

		# Save the cumulative saliency maps
		cm_sal_path = os.path.join(self.saliency_path, 'cumulative_full_saliency.npz')
		cropped_cm_sal_path = os.path.join(self.saliency_path, 'cropped_cumulative_full_saliency.npz')

		np.savez(cm_sal_path, data=cumulative_saliency)
		np.savez(cropped_cm_sal_path, data=cropped_cumulative_saliency)

	# Code to prepare weighted datasets
	def prepare_weights(self):

		data_df = pd.read_csv(os.path.join(self.data_path, 'normalized.csv'))
		data_np = data_df.to_numpy()

		saliency_path = os.path.join(self.saliency_path, 'cropped_cumulative_full_saliency.npz')

		weighted_dataset_path = os.path.join(self.data_path, 'weighted.csv')
		derive_weights(data_np, saliency_path, weighted_dataset_path)   

	def generate_density_map(self):

		data_df = pd.read_csv(os.path.join(self.data_path, 'weighted.csv'))
		full_data = data_df.to_numpy()

		# Compute density scores using Kernel Density estimation for all datasets but hidden_corr
		if self.name == 'hidden_corr':

			# Compute image-based density-map - pick smallest <marker_size, transparency>
			image_based_density_weights(full_data, 
										[-0.05, 1.05], 
										[-0.05, 1.05],
										img_name=f'{self.cm_saliency_img_path}/{self.name}_alpha_{self.transparency_levels[-1]}_marker_{self.marker_levels[-1]}.png',
										density_image_name=f'{self.saliency_path}/image_density_map',
										blurred_density_filename=f'{self.saliency_path}/image_blurred_density_map.png',
										map_name=f'{self.saliency_path}/image_blurred_density_map.npz',
										m_size=self.marker_levels[0], 
										opacity=self.transparency_levels[0], 
										std=2)
			# Load the density_map & use it to assign weights
			dense_map = np.load(f'{self.saliency_path}/image_blurred_density_map.npz')['data']
			print(dense_map.shape)

			final_data = assign_density_weights(full_data, dense_map, dense_map.shape[1], dense_map.shape[0])

		else:
			data_np = full_data[:, :2]

			# Fit KDE
			kde = KernelDensity(kernel=self.kernel, bandwidth=self.kernel_bandwidth).fit(data_np)

			# Densities scores are real numbers here 
			densities = kde.score_samples(data_np)

			# Perform Max-Min normalization to map density weights to a [0, 1] range. 
			# To avoid having zero weights for density we add a small constant factor to give every point some chance of being selected  
			densities_normalized = (densities - np.min(densities)) / (np.max(densities) - np.min(densities)-1e-6) + 1e-2
			densities_normalized = np.clip(densities_normalized, 0, 1) 

			# Store the fully-weighted version of the dataset
			final_data = np.hstack((full_data, densities_normalized.reshape(densities_normalized.shape[0], 1)))

		df = pd.DataFrame(final_data, columns=['X', 'Y', 'saliency', 'density'])
		df.to_csv(os.path.join(self.data_path, 'saliency_and_density_weights.csv'), index=False)

	# Code to plot different visual stimuli for a given dataset 
	def generate_subplots(self, alpha_values, marker_sizes, ptype='samples', colormap=plt.cm.gray):

		A = 10
		B = A / 1.25
		fig, axs = plt.subplots(len(alpha_values), len(marker_sizes), figsize=(A/len(marker_sizes)*len(marker_sizes), B/len(alpha_values)*len(alpha_values)))

		names = []

		for cc in range(len(marker_sizes)):
			names.append(f'Size: {marker_sizes[cc]}')

		for i in range(len(alpha_values)): 

			TR = alpha_values[i]

			for j in range(len(marker_sizes)):
				
				ms = marker_sizes[j]
				#print(ms, TR)

				if ptype == 'saliency_maps':
					saliency_map = np.load(os.path.join(self.cm_saliency_map_path, f'full_saliency_alpha_{TR}_marker_{ms}.npz'))['data']
					axs[i, j].imshow(saliency_map, cmap=colormap)

				elif ptype == 'samples':
					# Read sample data
					pd_frame = pd.read_csv(os.path.join(self.data_path, 'normalized.csv'))
					data_np = pd_frame.to_numpy()

					axs[i, j].scatter(data_np[:, 0], data_np[:, 1], color=self.color, s=ms,  alpha=TR)
					axs[i, j].set_xlim([-0.05, 1.05])
					axs[i, j].set_ylim([-0.05, 1.05])

				axs[i, j].set_xticks([])
				axs[i, j].set_yticks([])    

				if j == 0:
					axs[i, j].set_ylabel(f'Opacity: {TR}', fontsize=15, fontname='Palatino Linotype')

		for ax, name in zip(axs[0], names):
			ax.set_title(name, fontsize=15, fontname='Palatino Linotype')     

		fig.tight_layout()
		plt.savefig(os.path.join(self.saliency_path, f'visual_stimuli_{ptype}.pdf'))

###############################################################
######################## DATASETS #############################
###############################################################
class HiddenCorrelation(Dataset):

	def __init__(self, experiment_settings):

		name = 'hidden_corr'

		data_path = f'{name}/data'
		saliency_path = f'{name}/saliency_data'
		sampling_path = f'{name}/sampling_techniques'

		attr_x = 'x'
		attr_y = 'y'

		size = -1
		epsilon = 1e-2

		super().__init__(data_path, saliency_path, sampling_path, attr_x, attr_y, size, name, epsilon, experiment_settings)

class EstateCorrelation(Dataset):

	def __init__(self, experiment_settings):

		name = 'estate_corr'

		data_path = f'{name}/data'
		saliency_path = f'{name}/saliency_data'
		sampling_path = f'{name}/sampling_techniques'

		attr_x = 'number_of_reviews'
		attr_y = 'reviews_per_month'

		size = -1
		
		epsilon = 1e-2

		super().__init__(data_path, saliency_path, sampling_path, attr_x, attr_y, size, name, epsilon, experiment_settings)

# Epileptic Seizure #       
class EpilepticCorrelation(Dataset):

	def __init__(self, experiment_settings):

		name = 'epileptic_corr'

		data_path = f'{name}/data'
		saliency_path = f'{name}/saliency_data'
		sampling_path = f'{name}/sampling_techniques'

		attr_x = 'X1'
		attr_y = 'X11'

		size = -1
		epsilon = 1e-2

		super().__init__(data_path, saliency_path, sampling_path, attr_x, attr_y, size, name, epsilon, experiment_settings)

# Estate GIS # 
class EstateAnomalies(Dataset):

	def __init__(self, experiment_settings):

		name = 'estate_anomalies'

		data_path = f'{name}/data'
		saliency_path = f'{name}/saliency_data'
		sampling_path = f'{name}/sampling_techniques'

		attr_x = 'latitude'
		attr_y = 'longitude'

		size = -1
		epsilon = 1e-3
		super().__init__(data_path, saliency_path, sampling_path, attr_x, attr_y, size, name, epsilon, experiment_settings)

class ACSI(Dataset):

	def __init__(self, experiment_settings):

		name = 'acsi'

		data_path = f'{name}/data'
		saliency_path = f'{name}/saliency_data'
		sampling_path = f'{name}/sampling_techniques'

		attr_x = 'X2'
		attr_y = 'X6'

		size = -1
		epsilon = 1e-3

		super().__init__(data_path, saliency_path, sampling_path, attr_x, attr_y, size, name, epsilon, experiment_settings)

class MNIST(Dataset):

	def __init__(self, experiment_settings):

		name = 'mnist'

		data_path = f'{name}/data'
		saliency_path = f'{name}/saliency_data'
		sampling_path = f'{name}/sampling_techniques'

		attr_x = 'X'
		attr_y = 'Y'

		size = -1
		epsilon = 1e-2

		super().__init__(data_path, saliency_path, sampling_path, attr_x, attr_y, size, name, epsilon, experiment_settings)

class Intro(Dataset):

	def __init__(self, experiment_settings):

		name = 'intro'

		data_path = f'{name}/data'
		saliency_path = f'{name}/saliency_data'
		sampling_path = f'{name}/sampling_techniques'

		attr_x = 'X'
		attr_y = 'Y'

		size = -1
		epsilon = 1e-2

		super().__init__(data_path, saliency_path, sampling_path, attr_x, attr_y, size, name, epsilon, experiment_settings)     
###############################################################         



###############################################################
# Functions for perception weight assignments
###############################################################
def assign_weights_resolution(data_np, compressed_saliency, num_bins_x, num_bins_y):
	start = time.time()

	final_data = []

	# Apply a grid on the data
	H, xedges, yedges = np.histogram2d(data_np[:, 0], data_np[:, 1], bins=[num_bins_x, num_bins_y])

	# How to map points to bins: stackoverflow post
	# https://stackoverflow.com/questions/35961944/retrieving-bin-data-from-2d-histograms-in-numpy
	xedges[-1] += 1e-4 # add a small epsilon value to guarantee that the point with xedges[-1] falls into the last bucket
	
	# Reverse yedges to be sorted in decreasing order to follow cartesian structure
	yedges = yedges[::-1]

	# move the lowest boundary value to guarantee that the point with xedges[-1] falls into the last bucket
	yedges[-1] -= 1e-4 

	# https://numpy.org/doc/stable/reference/generated/numpy.digitize.html
	# Bins start from [1, |num_bins_x|]
	# -1 is needed to access the correct entries in the saliency matrix
	x_assignment = np.digitize(data_np[:, 0], xedges) -1 
	y_assignment = np.digitize(data_np[:, 1], yedges, right=True) -1

	# Find all the positions where x_assignment > num_bins_x, and reduce it by -1
	x_positions = np.where(x_assignment > num_bins_x)[0]
	x_assignment[x_positions] -= 1

	y_positions = np.where(y_assignment > num_bins_y)[0]
	y_assignment[y_positions] -= 1

	final_assignment = list(zip(y_assignment, x_assignment))
	final_assignment_np = np.asarray(final_assignment)

	# Iterate over the points
	for i in range(len(final_assignment_np)):
		cur_weight = compressed_saliency[final_assignment_np[i, 0], final_assignment_np[i, 1]]
		final_data.append([data_np[i, 0], data_np[i, 1], cur_weight])

	end = time.time()

	return final_data

def derive_weights(data_np, path_to_saliency, data_path):

	saliency_map = np.load(path_to_saliency)['data']
	final_data = assign_weights_resolution(data_np, saliency_map, saliency_map.shape[1], saliency_map.shape[0])

	# Store the dataset
	df = pd.DataFrame(final_data, columns=['X', 'Y', 'saliency'])
	sorted_df = df.sort_values(by=['X', 'Y'], ascending=[True, False])
	sorted_df.to_csv(data_path, index=False)

def image_based_density_weights(data, 
								x_axis, 
								y_axis,
								img_name, 
								density_image_name, 
								blurred_density_filename, 
								map_name, 
								m_size=2, 
								opacity=0.1, 
								std=2):

	def normalize_map(my_map, epsilon=1e-6, delta_boost=1e-2):
		my_map = (my_map - np.min(my_map)) / (np.max(my_map)-np.min(my_map)-epsilon) + delta_boost
		return my_map

	# Plot the density map
	custom_plot_function(data, 
						 x_axis, 
						 y_axis, 
						 density_image_name,
						 background_color='black', 
						 my_color='white',
						 marker_size=m_size, 
						 t=opacity)
	
	# Read the image
	density_perception_map = cv2.imread(density_image_name+'.png')
	density_perception_map = cv2.cvtColor(density_perception_map, cv2.COLOR_BGR2GRAY)

	density_perception_map = normalize_map(density_perception_map)
	
	# Keep the same area of image as we did for saliency
	# Crop the density map 
	eng = matlab.engine.start_matlab()
	cropped_columns, cropped_rows = eng.crop_image(img_name, nargout=2)
	
	data_columns = np.asarray(cropped_columns[0]).astype('int')
	data_columns = data_columns.reshape((data_columns.shape[1],))
	data_rows = np.asarray(cropped_rows[0]).astype('int')
	data_rows = data_rows.reshape((data_rows.shape[1],))
	
	density_perception_map = density_perception_map[data_rows[0]:data_rows[-1]+1, data_columns[0]:data_columns[-1]+1]

	# Blur the density map
	blurred_density_perception_map = gaussian_filter(density_perception_map, sigma=std)
	
	# Normalize the density map into [0, 1] range
	blurred_density_perception_map = normalize_map(blurred_density_perception_map)

	# Save the the blurred density map as png
	plt.figure(figsize=(14, 12))
	plt.imshow(blurred_density_perception_map, cmap=plt.cm.viridis)
	plt.axis('off')
	#plt.tight_layout(pad=0, h_pad=0, w_pad=0)
	plt.savefig(blurred_density_filename, bbox_inches='tight', pad_inches = 0)
	plt.show()
	
	np.savez(map_name, data=blurred_density_perception_map)
	
	eng.quit()

def assign_density_weights(data_np, density_map, num_bins_x, num_bins_y, epsilon=1e-6, delta_boost=1e-2):
	
	final_data = []
	
	print(data_np.shape)
	
	# Apply a grid on the data
	H, xedges, yedges = np.histogram2d(data_np[:, 0], data_np[:, 1], bins=[num_bins_x, num_bins_y])

	# How to map points to bins: stackoverflow post
	# https://stackoverflow.com/questions/35961944/retrieving-bin-data-from-2d-histograms-in-numpy
	xedges[-1] += 1e-4 # add a small epsilon value to guarantee that the point with xedges[-1] falls into the last bucket
	
	# Reverse yedges to be sorted in decreasing order to follow cartesian structure
	yedges = yedges[::-1]

	# move the lowest boundary value to guarantee that the point with xedges[-1] falls into the last bucket
	yedges[-1] -= 1e-4 

	# https://numpy.org/doc/stable/reference/generated/numpy.digitize.html
	# Bins start from [1, |num_bins_x|]
	# -1 is needed to access the correct entries in the saliency matrix
	x_assignment = np.digitize(data_np[:, 0], xedges) -1 
	y_assignment = np.digitize(data_np[:, 1], yedges, right=True) -1

	# Find all the positions where x_assignment > num_bins_x, and reduce it by -1
	x_positions = np.where(x_assignment > num_bins_x)[0]
	x_assignment[x_positions] -= 1

	y_positions = np.where(y_assignment > num_bins_y)[0]
	y_assignment[y_positions] -= 1

	final_assignment = list(zip(y_assignment, x_assignment))
	final_assignment_np = np.asarray(final_assignment)

	# Iterate over the points
	for i in range(len(final_assignment_np)):
		cur_weight = density_map[final_assignment_np[i, 0], final_assignment_np[i, 1]]
		final_data.append([data_np[i, 0], data_np[i, 1], data_np[i, 2], cur_weight])

	density_weighted_data = np.array(final_data)

	#Max-Min normalization with some boost
	density_weighted_data[:, 3] = (density_weighted_data[:, 3] - np.min(density_weighted_data[:, 3])) / (np.max(density_weighted_data[:, 3])-np.min(density_weighted_data[:, 3]) - epsilon) + delta_boost
	density_weighted_data[:, 3] = np.clip(density_weighted_data[:, 3], 0, 1) 
	
	return density_weighted_data

###############################################################



###############################################################
# Functions to handle data preparation
###############################################################
def initialize_dataloader(dataset_name, plot_setting):

	# @-todo: Add code to support more datasets
	if dataset_name == 'epileptic_corr':
		dataset = EpilepticCorrelation(plot_setting)

	elif dataset_name == 'estate_anomalies':
		dataset = EstateAnomalies(plot_setting)

	elif dataset_name == 'estate_corr':
		dataset = EstateCorrelation(plot_setting)

	elif dataset_name == 'mnist':
		dataset = MNIST(plot_setting)

	elif dataset_name == 'hidden_corr':
		dataset = HiddenCorrelation(plot_setting)

	elif dataset_name == 'acsi':
		dataset = ACSI(plot_setting)

	elif dataset_name == 'intro':
		dataset = Intro(plot_setting)

	return dataset  

# Code to preoare stοrage paths for a given dataset
def prepare_storage_paths(dataset_name):

	# Use dummy settings to initialize the objects
	plot_setting = {}
	plot_setting['x_canvas_size'] = 14 #18
	plot_setting['y_canvas_size'] = 12
	plot_setting['dpi_value'] = 100
	plot_setting['marker'] = 1
	plot_setting['alpha'] = 0.1
	plot_setting['x_lim'] = [-0.05, 1.05]
	plot_setting['y_lim'] = [-0.05, 1.05]
	plot_setting['color'] = 'maroon'
	plot_setting['background_color'] = 'white'

	plot_setting['resize_x'] = 800 
	plot_setting['resize_y'] = 640 

	transparency_values = np.arange(0.005, 0.3, 0.01)

	for i in range(transparency_values.shape[0]):
		transparency_values[i] = round(transparency_values[i], 3)

	marker_values = np.array([1])
		
	plot_setting['transparency_values'] = transparency_values
	plot_setting['marker_values'] = marker_values

	plot_setting['marker_level_start'] = 0
	plot_setting['marker_level_end'] = marker_values.shape[0] 

	plot_setting['transparency_level_start'] = 0
	plot_setting['transparency_level_end'] = transparency_values.shape[0]

	plot_setting['kernel'] = 'tophat'
	plot_setting['kernel_bandwidth'] = 0.02

	dataset = initialize_dataloader(dataset_name, plot_setting)     
	dataset.create_storage_paths()  

# Code to generate the weighted version of a dataset
def prepare_weighted_dataset(dataset_name):

	print("Preparing weighted version of {} dataset".format(dataset_name))

	# Read the settings for the weighted scheme 
	with open(f'{dataset_name}/weighting_phase_setting.json') as json_data:
		setting = json.load(json_data) 

	dataset = initialize_dataloader(dataset_name, setting)

	"""
	dataset.normalize_data()

	# Step 1: Generate all the intermediate stimuli
	dataset.generate_intermediate_saliency_maps()

	# Step 2: Overlay & perform max operation to get the aggregate map
	dataset.overlay_saliency_maps()

	dataset.prepare_weights()
	dataset.plot_cumulative_saliency()
	dataset.plot_cropped_cumulative_saliency()
	"""
	
	# Step 3: Compute the density map 
	dataset.generate_density_map()

def plot_visual_stimuli(dataset_name, colormap=plt.cm.gray):

	# Read the specification for weighting scheme
	with open(f'{dataset_name}/weighting_phase_setting.json') as json_data:
		setting = json.load(json_data)

	dataset = initialize_dataloader(dataset_name, setting)
	dataset.generate_subplots(np.asarray(setting['transparency_values']), np.asarray(setting['marker_values']), 'samples')
	dataset.generate_subplots(np.asarray(setting['transparency_values']), np.asarray(setting['marker_values']), 'saliency_maps', colormap)

def plot_saliency_maps(dataset_name, saliency_type='cumulative', colormap=plt.cm.viridis):

	with open(f'{dataset_name}/weighting_phase_setting.json') as json_data:
		setting = json.load(json_data)

	dataset = initialize_dataloader(dataset_name, setting)
	dataset.plot_cumulative_saliency(cumulative=True, colormap=colormap)    
	dataset.plot_cropped_cumulative_saliency(cumulative=True, colormap=colormap)

def visualize_data_heatmap(dataset_name, map_type='saliency'):

	with open(f'{dataset_name}/weighting_phase_setting.json') as json_data:
		setting = json.load(json_data)

	dataset = initialize_dataloader(dataset_name, setting)
	print(dataset.data_path)

	full_data = pd.read_csv(os.path.join(dataset.data_path, 'saliency_and_density_weights.csv')).to_numpy()

	# Read weighted dataset
	if map_type == 'saliency':
		weighted_data = np.hstack((full_data[:, :2], full_data[:, 2].reshape(full_data[:, 2].shape[0], 1)))
	else:
		weighted_data = np.hstack((full_data[:, :2], full_data[:, 3].reshape(full_data[:, 2].shape[0], 1)))

	# Plot heatmap
	custom_plot_function(weighted_data, [-0.05, 1.05], [-0.05, 1.05], filename=os.path.join(f'{dataset_name}/{map_type}_heatmap'), plot_type='heatmap', marker_size=8)
	
def empirical_density_computation(dataset_names):

	density_variance = {}

	for dataset_name in dataset_names:

		data_df = pd.read_csv(f'{dataset_name}/data/saliency_and_density_weights.csv')
		data_np = data_df.to_numpy()
	
		variance = np.var(np.unique(data_np[:, 3]))
		density_variance[dataset_name] = variance
		
	# Store the results into a json file
	with open(f"density_variance_info.json", "w") as outfile:
		json.dump(density_variance, outfile)	
