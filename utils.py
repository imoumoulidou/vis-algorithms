import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import os

from PIL import Image

from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Dictionaries for formatting the plots for each method
colors = {'random': '#60A216', 'max_min': '#028A82', 'vas': '#F3C30F', 'density_biased': '#E7298A', 'perception_aware': '#F29E1D', 'perception_aware_with_density': '#04247C',
		 'box_sampling_0.01_0.01': '#fee6ce', 'box_sampling_0.005_0.001': '#fdae6b', 'box_sampling_0.001_0.01':'#F83C31', 
		 'box_sampling_0.003_0.01': '#fee6ce', 'box_sampling_0.002_0.001': '#fdae6b',  
		 'box_sampling_0.001_0.001':'#e6550d',
		 'approximate': '#FF9307',
		 'blue_noise' :'#7570b3',
		 'vas_conv': '#937860'}

marker_styles = {'random': 'x', 'max_min': 'o', 'vas': '^', 'density_biased': 'h', 'perception_aware': 's',
				'box_sampling_0.01_0.01': 'x', 'box_sampling_0.005_0.001': 'o', 'box_sampling_0.001_0.01':'^', 
				'box_sampling_0.003_0.01': 'x', 'box_sampling_0.002_0.001': 'o',
				'box_sampling_0.001_0.001':'^', 
				'approximate': 'D',
				'blue_noise':'+',
				'vas_conv': 'p'}

marker_sizes = {'random': 8, 'max_min': 8, 'vas': 8, 'density_biased': 8, 'perception_aware': 14, 
				'box_sampling_0.01_0.01': 8, 'box_sampling_0.005_0.001': 8, 'box_sampling_0.001_0.01': 8, 
				'box_sampling_0.003_0.01': 8, 'box_sampling_0.002_0.001': 8, 
				'box_sampling_0.01_0.01_old': 8, 'box_sampling_0.005_0.001_old': 8, 'box_sampling_0.001_0.01_old': 8, 
				'box_sampling_0.001_0.001':8, 'approximate': 8, 'blue_noise': 8, 'vas_conv':8}

line_styles = {'random': '--', 'max_min': '-.', 'vas': '-.', 'density_biased': '--', 'perception_aware': '-', 'blue_noise':':', 'vas_conv': 'solid',
			   'box_sampling_0.01_0.01': 8, 'box_sampling_0.005_0.001': 8, 'box_sampling_0.001_0.001':8, 
			   'box_sampling_0.003_0.01': 8, 'box_sampling_0.002_0.001': 8, 'approximate': 8}

running_time_line_styles = {'random': '--', 'max_min': '-.', 'vas': '-.', 'density_biased': '--', 'perception_aware': '-', 'blue_noise':':', 'vas_conv': 'solid',
							'box_sampling_0.01_0.01': 8, 'box_sampling_0.005_0.001': 8, 
							'box_sampling_0.003_0.01': 8, 'box_sampling_0.002_0.001': 8, 'box_sampling_0.001_0.001':8, 'approximate': 8}

running_time_line_width = {'random': 2, 'max_min': 2, 'vas': 2, 'density_biased': 2, 'perception_aware': 2, 'blue_noise':2, 'vas_conv': 2}

line_width = {'random': 3, 'max_min': 3, 'vas': 3, 'density_biased': 3, 'perception_aware': 3, 'blue_noise':3}

marker_edge_width = {'random': 2, 'max_min': 2, 'vas': 2, 'density_biased': 2, 'perception_aware': 2, 
					'box_sampling_0.01_0.01': 2, 'box_sampling_0.005_0.001': 2, 'box_sampling_0.001_0.01': 2,
					'box_sampling_0.003_0.01': 2, 'box_sampling_0.002_0.001': 2,
					'box_sampling_0.01_0.01_old': 2, 'box_sampling_0.005_0.001_old': 2, 'box_sampling_0.001_0.01_old': 2,
					'box_sampling_0.001_0.001':2, 'approximate': 2, 'blue_noise':2, 'vas_conv': 2,}

marker_facecolor = {'random': '#60A216', 'max_min': 'none' , 'vas': '#F3C30F', 'density_biased': 'none', 'perception_aware': '#04247C',
					'box_sampling_0.01_0.01': '#76B947', 'box_sampling_0.005_0.001': '#CAC12E', 'box_sampling_0.001_0.01':'#F83C31',
					'box_sampling_0.003_0.01': '#76B947', 'box_sampling_0.002_0.001': '#CAC12E',  
					'box_sampling_0.01_0.01_old': '#76B947', 'box_sampling_0.005_0.001_old': '#CAC12E', 'box_sampling_0.001_0.01_old':'#F83C31',
					'box_sampling_0.001_0.001':'#F83C31',
					'approximate': '#FF9307', 'blue_noise':'k', 'vas_conv': '#937860'}

legend_info = {'random': 'Random', 'max_min': 'Max-Min', 'vas': 'VAS', 'density_biased': 'DBS', 'perception_aware': 'PAwS',
			   'box_sampling_0.01_0.01': 'High Compression', 'box_sampling_0.005_0.001': 'Medium Compression', 'box_sampling_0.001_0.01': 'Low Compression', 
			   'box_sampling_0.003_0.01': 'High Compression', 'box_sampling_0.002_0.001': 'Medium Compression',
			   'box_sampling_0.01_0.01_old': 'High Compression', 'box_sampling_0.005_0.001_old': 'Med Compression', 'box_sampling_0.001_0.01_old': 'Low Compression',
			   'box_sampling_0.001_0.001':'Low Compression', 'approximate': 'ApproPAwS', 'blue_noise': 'Blue-Noise',
			   'vas_conv': 'VAS (Conv.)'}

# Stackoverflow: https://stackoverflow.com/questions/1995373/deleting-all-files-in-a-directory-with-python
def clear_folder(path_to_folder):
	
	# List all files in directory
	for f in os.listdir(path_to_folder):
		os.remove(os.path.join(path_to_folder, f))

def resize_image(img1_path, img2_path, width=800, height=640):

	# Resize the density map
    img = Image.open(img1_path)
    new_image = img.resize((width, height))
    new_image.save(img2_path)

def max_min_normalization(data_np):

	data_np = (data_np - np.min(data_np)) / (np.max(data_np)-np.min(data_np))
	
	return data_np
    
def custom_plot_function(data, x_limits, y_limits, 
				  			filename, dpi_value=100, plot_type='scatter', 
                  				savefig=True, figtype='png', color_map=plt.cm.viridis, 
                  					background_color='white', my_color='maroon', remove_axis='True',
                  						marker_size=1, t=1, x_axis_size=14, y_axis_size=12):
	
	if plot_type == 'scatter':
		plt.figure(figsize=(x_axis_size, y_axis_size), facecolor=background_color)
		plt.scatter(data[:, 0], data[:, 1], color=my_color, s=marker_size, alpha=t)
		
		# Set limits for the axes
		plt.xlim([x_limits[0], x_limits[1]])
		plt.ylim([y_limits[0], y_limits[1]])

	elif plot_type == 'heatmap':
		plt.figure(figsize=(x_axis_size, y_axis_size), facecolor=background_color)
		plt.scatter(data[:, 0], data[:, 1], c=data[:, 2], s=marker_size, alpha=t, cmap=color_map)

		# Set limits for the axes
		plt.xlim([x_limits[0], x_limits[1]])
		plt.ylim([y_limits[0], y_limits[1]])

	elif plot_type == 'image':
		plt.figure(figsize=(x_axis_size, y_axis_size))
		plt.imshow(data, cmap=color_map, vmin=0, vmax=1)	

	if remove_axis:	
		plt.axis('off')

	filename = f'{filename}.{figtype}'
	plt.savefig(f'{filename}', dpi=dpi_value, bbox_inches='tight', pad_inches = 0)
	
	plt.close()

# Code to plot different partitioning schemes for a dataset
def partitioning_subplot(dataset, approx_thresholds, saliency_thresholds, seed, filename, 
							x_limits=(-0.05, 1.05), y_limits=(-0.05, 1.05), color_map= plt.cm.viridis, savefig=True, xsize=5, ysize=5):

	fig, axs = plt.subplots(nrows=len(approx_thresholds), ncols=len(saliency_thresholds), figsize=(xsize, ysize))

	count = 0 

	for i in range(len(approx_thresholds)):

		for j in range(len(saliency_thresholds)):

			approx_threshold = approx_thresholds[i]
			saliency_threshold = saliency_thresholds[j]

			# Load the information about the partitioning scheme
			filepath = os.path.join(f'{dataset}/sampling_techniques/box_sampling_{approx_threshold}_{saliency_threshold}/cumulative_saliency/data/approximate_scheme.npz')
			info = np.load(filepath, allow_pickle=True)

			coordinates_per_box = info['coordinates_per_box'] 
			average_saliency_per_box = info['average_saliency_per_box']

			# Plot the boxes
			for cc in range(len(coordinates_per_box)):

				splits = coordinates_per_box[cc]

				# Corner Orders: BL, BR, UL, UR
				rectangle = [(splits[0], splits[2]), (splits[1], splits[2]), (splits[0], splits[3]), (splits[1], splits[3])]

				xy = rectangle[0] 
				width = (splits[1] - splits[0]) #*scale_factors[count]  # xmax - xmin, rectangle[3][0] - rectangle[0][0]
				height = (splits[3] - splits[2]) #*scale_factors[count] # ymax-ymin, rectangle[3][1] - rectangle[0][1]

				rect = Rectangle(xy, width, height, linewidth=5, edgecolor='none', facecolor=color_map(average_saliency_per_box[cc]), alpha=1)
				axs[i, j].add_patch(rect)

				# Axis 
				axs[i, j].set(xlim=x_limits, ylim=y_limits)
				axs[i, j].set_xticks([])
				axs[i, j].set_yticks([])

				if j == 0:
					axs[i, j].set_ylabel(rf'$\lambda={round(approx_threshold, 4)}$', fontsize=20)

				if i == 0:
					axs[i, j].set_title(rf'$\sigma={round(saliency_threshold, 4)}$', fontsize=20)

			if dataset == 'acsi':
				actual_data = pd.read_csv(f'{dataset}/data/saliency_and_density_weights.csv').to_numpy()
				axs[i, j].scatter(actual_data[:, 0], actual_data[:, 1], color=color_map(average_saliency_per_box[cc]), s=0.5)

			if dataset != 'mnist' and dataset != 'hidden_corr' and dataset != 'acsi' and dataset != 'estate_corr':
				axs[i, j].text(0.03, 0, f'# boxes={len(coordinates_per_box)}', fontsize=22, fontfamily='Palatino Linotype')
			elif dataset == 'hidden_corr':
				axs[i, j].text(0.03, 0, f'# boxes={len(coordinates_per_box)}', fontsize=22, fontfamily='Palatino Linotype')#
			else: # mnist & acsi & estate_corr
				axs[i, j].text(0.03, 0.8, f'# boxes={len(coordinates_per_box)}', fontsize=22, fontfamily='Palatino Linotype')

			count += 1

	plt.tight_layout()

	print("Saving figure")

	if savefig:
		plt.savefig(filename, dpi=200, bbox_inches='tight')	

# Code to plot different partitioning schemes for a dataset
def partitioning_subplot_with_data(dataset, approx_thresholds, saliency_thresholds, seed, filename, 
										x_limits=(-0.05, 1.05), y_limits=(-0.05, 1.05), color_map= plt.cm.viridis, savefig=True, xsize=5, ysize=5):

	fig, axs = plt.subplots(nrows=len(approx_thresholds), ncols=len(saliency_thresholds), figsize=(xsize, ysize))

	letters = ['(1)', '(2)', '(3)', '(4)', '(5)', '(6)', '(7)', '(8)', '(9)']
	print(len(letters))

	count = 0 

	for i in range(len(approx_thresholds)):

		for j in range(len(saliency_thresholds)):

			approx_threshold = approx_thresholds[i]
			saliency_threshold = saliency_thresholds[j]

			# Load the information about the partitioning scheme
			filepath = os.path.join(f'{dataset}/sampling_techniques/box_sampling_{approx_threshold}_{saliency_threshold}/cumulative_saliency/data/approximate_scheme_with_data_per_box.npz')
			info =  np.load(filepath, allow_pickle=True)

			coordinates_per_box = info['coordinates_per_box'] 
			average_saliency_per_box = info['average_saliency_per_box']
			data_per_box = info['data_per_box']

			# Plot the boxes
			for cc in range(len(coordinates_per_box)):

				# Retrieve the actual data points
				rectangle_points = data_per_box[cc]
				splits = coordinates_per_box[cc]

				# Corner Orders: BL, BR, UL, UR
				rectangle = [(splits[0], splits[2]), (splits[1], splits[2]), (splits[0], splits[3]), (splits[1], splits[3])]

				xy = rectangle[0] 
				width = (splits[1] - splits[0])#*scale_factors[count]  # xmax - xmin, rectangle[3][0] - rectangle[0][0]
				height = (splits[3] - splits[2])#*scale_factors[count] # ymax-ymin, rectangle[3][1] - rectangle[0][1]

				rect = Rectangle(xy, width, height, linewidth=5, edgecolor='none', facecolor=color_map(average_saliency_per_box[cc]), alpha=1)

				# Plot data for small rectangles with width close to zero
				if width*height<=1e-4: # dataset == 'acsi' and  width*height<= 1e-3 
					axs[i, j].scatter(rectangle_points[:, 0], rectangle_points[:, 1], c=[color_map(average_saliency_per_box[cc])]*len(rectangle_points[:, 0]), s=0.2, alpha=0.7)
				
				#if dataset == 'acsi':
					#print("adding points onto the plot")
				#	axs[i, j].scatter(rectangle_points[:, 0], rectangle_points[:, 1], c=[color_map(average_saliency_per_box[cc])]*len(rectangle_points[:, 0]), s=1, alpha=0.7)

				axs[i, j].add_patch(rect)

				# Axis 
				axs[i, j].set(xlim=x_limits, ylim=y_limits)
				axs[i, j].set_xticks([])
				axs[i, j].set_yticks([])

				if j == 0:
					axs[i, j].set_ylabel(rf'$\lambda={round(approx_threshold, 3)}$', fontsize=20)

				if i == 0:
					axs[i, j].set_title(rf'$\sigma={round(saliency_threshold, 4)}$', fontsize=20)

			if dataset != 'mnist' and dataset != 'hidden_corr' and dataset != 'acsi' and dataset != 'estate_corr':
				axs[i, j].text(0.03, 0, f'# boxes={len(coordinates_per_box)}', fontsize=22, fontfamily='Palatino Linotype')
			elif dataset == 'hidden_corr':
				#print("boo")
				axs[i, j].text(0.03, 0, f'# boxes={len(coordinates_per_box)}', fontsize=22, fontfamily='Palatino Linotype')#
			else: # mnist & acsi & estate_corr
				axs[i, j].text(0.03, 0.8, f'# boxes={len(coordinates_per_box)}', fontsize=22, fontfamily='Palatino Linotype')
			count += 1

	plt.tight_layout()

	print("Saving figure")

	if savefig:
		plt.savefig(filename, dpi=200, bbox_inches='tight')	


# Code to plot different the qualitative results of partitioning schemes for a dataset
def partitions_qual_subplot(dataset, approx_thresholds, saliency_thresholds, seed, filename, 
								x_limits=(-0.05, 1.05), y_limits=(-0.05, 1.05), color_map= plt.cm.viridis, savefig=True, xsize=5, ysize=5, sample_size=9611):

	fig, axs = plt.subplots(nrows=len(approx_thresholds), ncols=len(saliency_thresholds), figsize=(xsize, ysize))

	count = 0 

	for i in range(len(approx_thresholds)):

		for j in range(len(saliency_thresholds)):

			approx_threshold = approx_thresholds[i]
			saliency_threshold = saliency_thresholds[j]

			# Load the sample
			filepath = os.path.join(f'{dataset}/sampling_techniques/box_sampling_{approx_threshold}_{saliency_threshold}/cumulative_saliency/data/repetition_{seed}/box_sampling_{approx_threshold}_{saliency_threshold}_{sample_size}.npz')
			sample =  np.load(filepath)['data']

			axs[i, j].scatter(sample[:, 0], sample[:, 1], c='#000C66', s=0.1, alpha=0.5)

			axs[i, j].set_xticks([])
			axs[i, j].set_xticklabels([])
			axs[i, j].set_yticks([])
			axs[i, j].set_yticklabels([])
			
			axs[i, j].set_ylim([-0.05, 1.05])
			axs[i, j].set_xlim([-0.05, 1.05])

			if j == 0:
				axs[i, j].set_ylabel(rf'$\lambda={round(approx_threshold, 3)}$', fontsize=20)

			if i == 0:
				axs[i, j].set_title(rf'$\sigma={round(saliency_threshold, 4)}$', fontsize=20)

			count += 1

	plt.tight_layout()

	print("Saving figure")

	if savefig:
		plt.savefig(filename, dpi=200, bbox_inches='tight')

# Code to plot different partitioning schemes for a dataset
def partitioning_subplot_v2(dataset, approx_thresholds, saliency_thresholds, seed, filename, 
							x_limits=(-0.05, 1.05), y_limits=(-0.05, 1.05), color_map= plt.cm.viridis, savefig=True, xsize=5, ysize=5):

	fig, axs = plt.subplots(nrows=len(approx_thresholds), ncols=len(saliency_thresholds), figsize=(xsize, ysize))
	count = 0 

	for i in range(len(approx_thresholds)):

		for j in range(len(saliency_thresholds)):

			approx_threshold = approx_thresholds[i]
			saliency_threshold = saliency_thresholds[j]

			# Load the information about the partitioning scheme
			filepath = os.path.join(f'{dataset}/sampling_techniques/box_sampling_{approx_threshold}_{saliency_threshold}/cumulative_saliency/data/approximate_scheme.npz')
			info =  np.load(filepath, allow_pickle=True)

			coordinates_per_box = info['coordinates_per_box'] 
			average_saliency_per_box = info['average_saliency_per_box']

			# Plot the boxes
			for cc in range(len(coordinates_per_box)):

				# Retrieve the actual data points
				splits = coordinates_per_box[cc]

				# Corner Orders: BL, BR, UL, UR
				rectangle = [(splits[0], splits[2]), (splits[1], splits[2]), (splits[0], splits[3]), (splits[1], splits[3])]

				xy = rectangle[0] 
				width = (splits[1] - splits[0])#*scale_factors[count]  # xmax - xmin, rectangle[3][0] - rectangle[0][0]
				height = (splits[3] - splits[2])#*scale_factors[count] # ymax-ymin, rectangle[3][1] - rectangle[0][1]

				rect = Rectangle(xy, width, height, linewidth=5, edgecolor='none', facecolor=color_map(average_saliency_per_box[cc]), alpha=1)
				axs[i, j].add_patch(rect)

				# Axis 
				axs[i, j].set(xlim=x_limits, ylim=y_limits)
				axs[i, j].set_xticks([])
				axs[i, j].set_yticks([])
				
			count += 1

	plt.tight_layout()

	print("Saving figure")

	if savefig:
		plt.savefig(filename, dpi=200, bbox_inches='tight')
