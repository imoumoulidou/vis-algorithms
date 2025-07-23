function [cont_x_subspace, cont_y_subspace] = crop_image(path)

	background_thres = 200;
	% Read img
	img = imread(path);

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

end
