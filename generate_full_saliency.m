function heatmap = saliency_func(path) 

	% Read img
	img = imread(path);

	% Compute saliency
	map = dvs(img);

	% Extract heatmap
	heatmap = map.master_map_resized;
	
end
