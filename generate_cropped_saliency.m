function cropped_heatmap = cropped_saliency(path) 

	background_thres = 200;

	% Read img
	img = imread(path);

	% Compute saliency
	map = dvs(img);

	% Extract heatmap
	heatmap = map.master_map_resized;

	%% Compress Heatmap to only the x-dimension
	heatMapMatrix = map.top_level_feat_maps{4}; % heatmap but in pixel intensity matrix
	grayScaleOriginal = rgb2gray(img); % transform original image (plotted line) to gray scale

	% Find the columns in x-axis that contain part of the image      
	x_axis_img = min(grayScaleOriginal); 

	% Retrieve columns that correspond to the actual image
	x_subspace = find(x_axis_img < background_thres);
	cont_x_subspace = [x_subspace(1):x_subspace(end)];

	y_axis_img = min(grayScaleOriginal, [], 2); % find min across rows now

	% Retrieve rows that correspond to the actual image
	y_subspace = find(y_axis_img < background_thres); % this is not necessarily consecutive indices
	cont_y_subspace = [y_subspace(1):y_subspace(end)];

	% Select the subspace of heatmap that corresponds to actual data points 
	cropped_heatmap = heatmap(cont_y_subspace, cont_x_subspace);

end
