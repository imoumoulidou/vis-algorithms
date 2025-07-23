import numpy as np
import pandas as pd
import os
import matlab.engine

from PIL import Image
from skimage.metrics import structural_similarity

import matplotlib.pyplot as plt 
import seaborn as sns
import json

from scipy.spatial import distance
from evaluation_metrics import *
from utils import *
from algorithms import *
import cv2

from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

class Experiment():

	def __init__(self, dataset, experimental_setting):

		self.dataset = dataset
		self.plot_setting = experimental_setting
		self.epsilon = experimental_setting['epsilon']

	def initialize_sampler(self, method):
		
		sampler = None

		if method == 'random':
			sampler = RandomSampler(self.dataset, self.plot_setting)
		
		elif method == 'max_min':
			sampler = MaxMinSampler(self.dataset, self.plot_setting)
		
		elif method == 'vas':
			sampler = VisualizationAwareSampler(self.dataset, self.plot_setting)

		elif method == 'perception_aware':
			sampler = PerceptionAwareSampler(self.dataset, self.plot_setting)

		elif method == 'perception_aware_with_density':
			sampler = PerceptionAwarewithDensitySampler(self.dataset, self.plot_setting)

		elif method == 'density_biased':
			sampler = DensityBiasedSampler(self.dataset, self.plot_setting)

		elif method == 'blue_noise':
			sampler = BlueNoiseSampler(self.dataset, self.plot_setting)

		else: # different box sampling algorithms
			sampler = BoxSampling(self.dataset, self.plot_setting, method)

		return sampler	

	def collect_samples(self, method, my_seeds, sample_sizes, return_time=False, save_sample=True):

		sampler = self.initialize_sampler(method)
		monitored_times = []

		for seed in my_seeds:

			for sample_size in sample_sizes:

				print("Testing {} for sample size {}".format(method, sample_size))
				sampler.create_storage_paths([seed])

				if method == 'random':
					sample_indices, sample, elapsed_time = sampler.compute_sample(sample_size, seed, True, save_sample=save_sample)
						
				elif method == 'max_min':
					sample_indices, sample, elapsed_time = sampler.compute_sample(sample_size, [], seed, True, save_sample=save_sample)

				elif method == 'perception_aware':
					sample_indices, sample, elapsed_time = sampler.compute_sample(sample_size, [], seed, True, save_sample=save_sample)

				elif method == 'perception_aware_with_density':
					sample_indices, sample, elapsed_time = sampler.compute_sample(sample_size, [], seed, True, save_sample=save_sample)	
				
				elif method == 'vas':
					sample_indices, sample, elapsed_time= sampler.compute_sample(sample_size, seed, True, save_sample=save_sample)

				elif method == 'density_biased':
					sample_indices, sample, elapsed_time = sampler.compute_sample(sample_size, seed, True, save_sample=save_sample)	
				
				elif method == 'blue_noise':
					sample_indices, sample, elapsed_time = sampler.compute_sample(sample_size, seed, True, save_sample=save_sample)

				else: # box sampling
					sample, elapsed_time = sampler.compute_sample(sample_size, seed, use_seed=True, save_sample=save_sample)

				monitored_times.append([elapsed_time, sample_size, method, seed])

				assert sample.shape[0] == sample_size, f"{method} did not match the correct sample size"

		if return_time:
			return monitored_times

	def generate_saliency_data(self, method, my_seeds, sample_sizes):

		sampler = self.initialize_sampler(method)

		for seed in my_seeds:

			for sample_size in sample_sizes:

				sampler.create_storage_paths([seed])

				# Step 1: Generate plots for all the intermediate ways to visualize a sample
				sampler.generate_intermediate_plots(sample_size, seed)	

				# Step 2: Generate saliency for all the intermediate ways to visualize a sample
				#sampler.generate_intermediate_saliency_maps(sample_size, seed)

				# Step 3: Plot intermediate saliency maps 
				#sampler.plot_intermediate_saliency_maps(sample_size, seed) 

				# Step 4: Compute Cumulative Saliency
				#sampler.overlay_intermediate_maps(sample_size, seed)

	# Collect data for all sampling techniques for a given dataset
	def run_experiment(self, methods, my_seeds, sample_sizes):

		for method in methods:

			self.collect_samples(method, my_seeds, sample_sizes)
			self.generate_saliency_data(method, my_seeds, sample_sizes)

	# Code to compute evaluation metrics across different metrics & sample sizes		
	def evaluate_methods(self, methods, metric, my_seeds, sample_sizes):

		metrics = [] 

		for method in methods:

			sampler = self.initialize_sampler(method)

			for seed in my_seeds:

				for sample_size in sample_sizes:
					
					stats = sampler.compute_metrics(sample_size, self.plot_setting['marker_values'], self.plot_setting['transparency_values'], seed, metric)

					raw_scores = stats['raw_scores']
					point_sizes = stats['point_sizes']
					opacity_percents = stats['opacity_percents']
					mean_score = stats['average']
					max_score = stats['max']
					min_score = stats['min']
					argmax = stats['argmax']
					argmin = stats['argmin']
					std = stats['std']

					for i in range(len(raw_scores)):
						metrics.append([raw_scores[i], point_sizes[i], opacity_percents[i], mean_score, max_score, min_score, std, argmax, argmin, sample_size, method, seed])	

		return metrics

	# Code to compute evaluation metrics for approximate PA across different metrics & sample sizes		
	def evaluate_approximate_methods(self, methods, metric, my_seeds, sample_sizes):

		metrics = [] 

		for method in methods:

			sampler = self.initialize_sampler(method)

			for seed in my_seeds:

				for sample_size in sample_sizes:

					stats = sampler.compute_metrics_for_approximate_technique(sample_size, self.plot_setting['marker_values'], self.plot_setting['transparency_values'], seed, metric)

					raw_scores = stats['raw_scores']
					point_sizes = stats['point_sizes']
					opacity_percents = stats['opacity_percents']
					mean_score = stats['average']
					max_score = stats['max']
					min_score = stats['min']
					argmax = stats['argmax']
					argmin = stats['argmin']
					std = stats['std']

					for i in range(len(raw_scores)):
						metrics.append([raw_scores[i], point_sizes[i], opacity_percents[i], mean_score, max_score, min_score, std, argmax, argmin, sample_size, method, seed])	

		return metrics

# Code to generate the samples & generate their saliency data
def perception_aware_pipeline(dataset_name, sampling_techniques, exp_seeds, sample_sizes):
	
	"""
	Inputs:
		-- dataset_name: a string corresponding to one of the datasets supported
		-- sampling_techniques: a list of methods supported
		-- exp_seeds: np.array or list of seeds for reproducibility
		-- sample_sizes: list of sample sizes
	"""
	
	with open(f'{dataset_name}/sampling_phase_setting.json') as json_data:
		setting = json.load(json_data)

	experiment = Experiment(dataset_name, setting)
	experiment.run_experiment(sampling_techniques, exp_seeds, sample_sizes)

# Code to generate only the samples
def produce_samples(dataset_name, sampling_technique, exp_seeds, sample_sizes):

	"""
	Inputs:
		-- dataset_name: a string corresponding to one of the datasets supported
		-- sampling_technique: a sampling method supported
		-- exp_seeds: np.array or list of seeds for reproducibility
		-- sample_sizes: list of sample sizes
	"""

	with open(f'{dataset_name}/sampling_phase_setting.json') as json_data:
		setting = json.load(json_data)

	experiment = Experiment(dataset_name, setting)
	experiment.collect_samples(sampling_technique, exp_seeds, sample_sizes)

# Code to generate only the saliency data
def produce_saliency_data(dataset_name, sampling_technique, exp_seeds, sample_sizes):

	with open(f'{dataset_name}/sampling_phase_setting.json') as json_data:
		setting = json.load(json_data)

	experiment = Experiment(dataset_name, setting)
	experiment.generate_saliency_data(sampling_technique, exp_seeds, sample_sizes)

# Code to retrieve data about running times
def compute_running_times(dataset_name, sampling_techniques, exp_seeds, sample_sizes, filename):

	collective_times = []
	
	with open(f'{dataset_name}/sampling_phase_setting.json') as json_data:
		setting = json.load(json_data)

	experiment = Experiment(dataset_name, setting)

	for sampling_technique in sampling_techniques:

		cur_times = experiment.collect_samples(sampling_technique, exp_seeds, sample_sizes, return_time=True, save_sample=False)	
		collective_times.extend(cur_times)

	# Store is a dataframe 
	df = pd.DataFrame(collective_times, columns=['time', 'size', 'sampling_method', 'seed'])
	df.to_csv(os.path.join(f'{dataset_name}/sampling_techniques/', filename), index=False)

# Code to collect metrics for different sampling methods
def collect_metrics(dataset_name, metrics, sampling_techniques, exp_seeds, sample_sizes, filenames):

	if not os.path.exists(f'{dataset_name}/sampling_techniques/evaluation_metrics'):
		os.makedirs(f'{dataset_name}/sampling_techniques/evaluation_metrics')

	with open(f'{dataset_name}/sampling_phase_setting.json') as json_data:
		setting = json.load(json_data)

	experiment = Experiment(dataset_name, setting)

	for i in range(len(metrics)):

		print("Collecting data for", metrics[i])
		results = experiment.evaluate_methods(sampling_techniques, metrics[i], exp_seeds, sample_sizes)

		df = pd.DataFrame(results, columns=['score', 'point_size', 'opacity', 'mean_score', 'max_score', 'min_score', 'std', 'argmax', 'argmin', 'size', 'sampling_method', 'seed'])
		df.to_csv(os.path.join(f'{dataset_name}/sampling_techniques/evaluation_metrics', filenames[i]), index=False)

# Code to collect metrics for different sampling methods
def collect_metrics_for_approximate_pa(dataset_name, metrics, sampling_techniques, exp_seeds, sample_sizes, filenames):

	if not os.path.exists(f'{dataset_name}/sampling_techniques/evaluation_metrics'):
		os.makedirs(f'{dataset_name}/sampling_techniques/evaluation_metrics')

	with open(f'{dataset_name}/sampling_phase_setting.json') as json_data:
		setting = json.load(json_data)

	experiment = Experiment(dataset_name, setting)

	for i in range(len(metrics)):
		
		results = experiment.evaluate_approximate_methods(sampling_techniques, metrics[i], exp_seeds, sample_sizes)

		df = pd.DataFrame(results, columns=['score', 'point_size', 'opacity', 'mean_score', 'max_score', 'min_score', 'std', 'argmax', 'argmin', 'size', 'sampling_method', 'seed'])
		df.to_csv(os.path.join(f'{dataset_name}/sampling_techniques/evaluation_metrics', filenames[i]), index=False)


###########################################################
# Code to experiment with different partitioning schemes
def box_sampling_parameters_experiment(dataset_name, seeds, approx_threshold, saliency_threshold):

	with open(f'{dataset_name}/weighting_phase_setting.json') as json_data:
		setting = json.load(json_data)

	for i in range(len(approx_threshold)):
		
		for j in range(len(saliency_threshold)):

			cur_approx_threshold = approx_threshold[i]
			cur_saliency_threshold = saliency_threshold[j]

			print("Experimenting with approx threshold", cur_approx_threshold, "saliency", cur_saliency_threshold)

			# Initialize the sampler 
			sampler = BoxSampling(dataset_name, setting, method=f'box_sampling_{cur_approx_threshold}_{cur_saliency_threshold}')
			sampler.create_storage_paths(seeds)

			# Retrieve the boxes
			group_boxes, group_statistics = sampler.retrieve_boxes_with_saliency(approximation_threshold=cur_approx_threshold, sal_threshold=cur_saliency_threshold, points=100)

			boxes_info, statistics_info, saliency_info, mean_saliency, boxes_sizes = sampler.aggregate_boxes_information(group_boxes, group_statistics)
			print("Number of boxes in compression:", len(boxes_sizes))
			
###########################################################