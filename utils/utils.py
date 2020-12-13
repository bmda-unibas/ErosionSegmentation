# Author: Maxim Samarin (maxim.samarin@unibas.ch)
# Last modification: 13.12.20
#
#

import os
import shutil
import cv2
import subprocess

try:
	import cPickle as pickle
except:
	import pickle

import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal, gdalnumeric
from PIL import Image

from utils.array2shapefile import array2shape

os.environ['QT_QPA_PLATFORM']='offscreen'


def normalise_images(images, max_val=np.inf, min_val=-np.inf):
	''' Normalises input images such that each entry is in the range [0,1].

	images: Images array of shape (number of images, image height, image width, channels)
	max_val: Maximum value in each image to be considered. All larger values are set / clipped to max_val
	min_val: Minimum value in each image to be considered. All smaller values are set / clipped to min_val
	'''

	norm_images = np.zeros(images.shape)
	num_images = images.shape[0]

	for img_count in range(num_images):
		tmp_image = images[img_count]
		tmp_image = np.clip(np.fabs(tmp_image), min_val, max_val)
		tmp_image = tmp_image - np.amin(tmp_image)
		tmp_image = tmp_image / np.amax(tmp_image)

		norm_images[img_count] = tmp_image

	return norm_images



def get_mask_matrix(input_image='manual', polygons='manual', output_dir='Output_erosion_classes_matrices', output_name='matrix'):
	'''	Obtain for an input image / raster layer the class label / mask for each pixel and save the matrix as a .csv file.

	input_image: Path to input image / raster layer. If input_image is not specified the manual settings below are used.
	polygons: Path to the shapefile providing the class assignment for each pixel
	output_dir: Output directory to save the .csv file
	output_name: Optional output name for output 'erosion_classes_"output_name".csv'
	'''

	R_script = os.path.join(os.getcwd(), 'utils', 'extract_mask.R')

	R_script_path = shutil.which("Rscript")

	# Call R script for extracting mask
	print("\n------- Calling R script 'extract_mask.R'")
	subprocess.call("{} --vanilla {} -i {} -p {} -o {} -n {}".format(R_script_path, R_script, input_image, polygons, output_dir, output_name), shell=True)




def split_train_test_dir(samples_path, num_row_tiles, num_col_tiles, left_index_of_test, output_path=''):
	'''	Split an image divided into tiles into a training and test region. It is expected that the tiles are enumerated from 1 to the total number of tiles
	and from left to right, top to bottom. left_index_of_test indicates the index of the upper-left tile of the test region.

	samples_path: Path to the folder containing the tiles
	num_row_tiles: Number of tiles in one row / in x direction
	num_col_tiles: Number of tiles in one row / in y direction
	left_index_of_test: Index of the upper-left tile of the test region
	output_path: The training/testing folders are created in samples_path. Optionally, an output folder for the two subfolders can be specified
	'''

	# Get file names
	sample_files = os.listdir(samples_path)
	sample_files.sort()

	# Absolute path to sample files
	sample_files = [os.path.join(samples_path, file) for file in sample_files]

	if output_path:
		# If output_path specified, save the splitted data in dedicated training and testing folders there
		samples_path = output_path

	# Create a directory for training and testing samples
	train_path = os.path.join(samples_path, 'Training')
	test_path = os.path.join(samples_path, 'Testing')

	if not os.path.exists(train_path) or not os.path.exists(test_path):
		os.makedirs(train_path)
		os.makedirs(test_path)

	# Create indices (starting from 0 instead of 1 -> thus left_index_of_test-1) from the index of upper-left tile to the end of the row
	indices_test = np.arange((left_index_of_test - 1) % num_row_tiles, num_row_tiles).reshape(1, -1)

	# Repeat this indices to the end of the columns. I.e. the extent of the test region is specified.
	# In the resulting matrix, all rows are the same.
	indices_test = np.repeat(indices_test, num_col_tiles - left_index_of_test // num_row_tiles, axis=0)

	# Obtain the increments to each entry in the previous matrix to get the correct indices for the test region
	indices_test_increments = (np.arange(left_index_of_test // num_row_tiles, num_col_tiles) * num_row_tiles).reshape(
		-1, 1)

	# Add the increments to obtain a matrix specifying the indices of the test region.
	indices_test = indices_test + indices_test_increments
	indices_test = indices_test.flatten()

	# Create matrix with all indices for all tiles and delete the indices which belong to the test region
	indices_training = np.arange(num_row_tiles * num_col_tiles)
	indices_training = np.delete(indices_training, indices_test)

	# Move files belonging to training and test region in the respective folder
	for ind in indices_training:
		shutil.move(sample_files[ind], train_path)

	for ind in indices_test:
		shutil.move(sample_files[ind], test_path)



def tile_image(image_matrix=None, input_path='manual', input_type='RGB', tiles_x=20, tiles_y=20, margin_size_x=0, margin_size_y=0, DEM_res_factor=8, save_tiles=True):
	''' Divide an input image into (possibly intersecting) tiles.
	If margin_size is specified, it is expected that the image tiles need to be cropped in the subsequent analysis/processing.
	With margin_size = 0 the input image is simply divided.

	image_matrix: Array for an input image of the shape [height, width, channels] (input_type = 'RGB') or [height, width]
				(input_type = 'DEM' or 'mask'). image_matrix does not need to be specified if input_path is given.
	input_path: Path to input image. If input_path and image_matrix are not specified the manual settings below are used.
				Please specify input_path if image_matrix is given.
	input_type: Specify whether the input is an RGB image -> "RGB", the corresponding classes mask -> "mask", or an DEM input -> "DEM"
	tiles_x: Number of tiles in x direction (horizontal)
	tiles_y: Number of tiles in y direction (vertical)
	margin_size_x: Pixel margin size which overlaps with other tiles in x direction (horizontal)
	margin_size_y: Pixel margin size which overlaps with other tiles in y direction (vertical))
	DEM_res_factor: Factor to adjust resolution of DEM map (coarser) to a potential RGB map (finer).
					I.e. for a DEM resolution of 2 m and RGB resolution 0.25 m, this would lead to DEM_res_factor = 8.
					Only required, if an DEM image is to processed and image_matrix is not specified (where it is expected to be performed before calling this function).
	save_tiles: Save image tiles to output folder
	'''

	maps = [input_type]

	# Optionally specify a reference image size to ensure that tiling is possible (can be left empty, too).
	# Specify size as [height, width] (in pixels)
	reference_image_size = np.array([])

	# Split tiles into training and test tiles if split_train_test = True
	# left_index_of_test is the index of the upper-left tile (enumerated according to img_enumerator) from where the splitting is performed
	split_train_test = False
	left_index_of_test = 18

	for input_map in maps:

		print("\n------- Tiles for", input_map)

		if save_tiles:
			# Create an output folder. Note that if the folder already exists, the script is stopped to prevent overwritting.
			image_name = os.path.splitext(os.path.basename(input_path))[0]
			# image_name = image_name
			os.makedirs(os.path.join("Output_Tiles",image_name))

		if image_matrix is None:
			# Load image in input_path as numpy array if image_matrix was not specified.
			# Otherwise it is expected that similar operations were performed before calling the "tile_image" function
			if input_map[0:3] == 'RGB':
				# Load the GeoTIFF
				image_matrix = gdalnumeric.LoadFile(input_path)

				if input_map.shape[0] == 4:
				# if input_map == 'RGB_Urs41_2004' or input_map == 'RGB_Urs41_2010':
					# For Urs_2004 leave out 4th band with only 255 values.
					# print('\nShape of RGB_Urs41_2004 image_matrix:', image_matrix.shape)
					image_matrix = image_matrix[:3]

				# Swap dimensions such that the shape is given by (height, width, 3)
				image_matrix = np.transpose(image_matrix, axes=(1, 2, 0))

			elif input_map[0:4] == 'mask':
				# If it is the mask matrix, load csv with numpy
				image_matrix = np.loadtxt(input_path)

			elif input_map[0:3] == 'DEM':
				# For DEM derivatives, adjust resolution factor.
				# A pixel from DEM with 2 m resolution contains
				# - 16 pixels (4 <=> DEM_res_factor) of aerial images with 0.5 m resolution.
				# - 64 pixels (8 <=> DEM_res_factor) of aerial images with 0.25 m resolution.
				image_matrix = gdalnumeric.LoadFile(input_path)

				image_matrix = np.repeat(image_matrix, DEM_res_factor, axis=0)
				image_matrix = np.repeat(image_matrix, DEM_res_factor, axis=1)

			else:
				# Stop script
				print("Problem with processing input_map =", input_map)
				return

		print('\nShape of image_matrix:', image_matrix.shape)

		if reference_image_size.size == 0:
			# If no reference_image_size was specified, set the size of the input_path
			reference_image_size = image_matrix.shape

		# For tiling the input_path obtain matrix row and column indices without outer margin
		indices_height = [i for i in range(margin_size_y, reference_image_size[0] - margin_size_y)]
		indices_width = [i for i in range(margin_size_x, reference_image_size[1] - margin_size_x)]

		if (len(indices_width) % tiles_x != 0) or (len(indices_height) % tiles_y != 0):
			# Stop script
			print('Error: The input image cannot be evenly divided into {0} x {1} tiles.'.format(tiles_y, tiles_x))
			return

		# Pixel height and width of a single tile
		tile_height = int(len(indices_height) / tiles_y)
		tile_width = int(len(indices_width) / tiles_x)

		# Obtain indices of tile borders
		indices_height = indices_height[::tile_height][1:]
		indices_width = indices_width[::tile_width][1:]

		# Repeat every border index twice to shift borders due to margins
		indices_height = np.repeat(indices_height, 2, axis=0)
		indices_width = np.repeat(indices_width, 2, axis=0)

		# Shift borders. For every tuple, subtract margin_size from left/upper border and add margin_size to right/lower border
		indices_height = indices_height - np.tile(np.array([-margin_size_y, margin_size_y]), tiles_y - 1)
		indices_width = indices_width - np.tile(np.array([-margin_size_x, margin_size_x]), tiles_x - 1)

		# Add borders for the first and last tiles in each row/column
		indices_height = np.concatenate([[0], indices_height, [reference_image_size[0]]]).reshape([tiles_y, 2])
		indices_width = np.concatenate([[0], indices_width, [reference_image_size[1]]]).reshape([tiles_x, 2])

		# Create empty array to save tiles which shall be returned by the function
		# if len(image_matrix.shape) == 3:
		# 	# RGB input_matrix with 3 channels at the end
		# 	tiles = np.empty((0, tile_height + 2*margin_size_y, tile_width + 2*margin_size_x, 3))
		# else:
		# 	# input_matrix with only one channel
		# 	tiles = np.empty((0, tile_height + 2*margin_size_y, tile_width + 2*margin_size_x))

		# Get total number of tiles to identify the number of digits to enumerate extracted tiles
		tot_num_tiles = tiles_x * tiles_y
		num_digits_tot = len(str(tot_num_tiles))
		img_enumerator = 1

		skipped_tiles = list()

		# Extract tiles from input_matrix
		for h in indices_height:
			for w in indices_width:
				tile = image_matrix[h[0]:h[1], w[0]:w[1]]

				# Pad integers with leading 0 s.t. sorting of file names becomes easier afterwards
				num_digits_height = len(str(reference_image_size[0]))
				num_digits_width = len(str(reference_image_size[1]))

				padded_h0 = str(h[0]).zfill(num_digits_height)
				padded_h1 = str(h[1]).zfill(num_digits_height)
				padded_w0 = str(w[0]).zfill(num_digits_width)
				padded_w1 = str(w[1]).zfill(num_digits_width)

				padded_img_enumerator = str(img_enumerator).zfill(num_digits_tot)

				if input_map[0:3] == 'RGB':

					if tile.dtype != 'uint8':
						# Sometimes it occurs that RGB images have higher pixel values than 255, but we want to restrict to 0-255 thus 'uint8'
						tile = tile.astype('uint8')

					# print(tile)

					# Calculate how many black pixels are contained in the image.
					# A large fraction of black pixels usually indicates that the tile is mostly outside of the actual image region (for non-rectangular input maps)
					tot_num_black_pixels = np.all(tile == 0, axis = -1).sum()

					# Special setting for colour_matched maps
					tot_num_black_pixels = np.all(tile == [2,2,0], axis = -1).sum()
					tot_num_pixels = (tile_height + 2*margin_size_y)*(tile_width + 2*margin_size_x)

					frac_black_pixel = tot_num_black_pixels/tot_num_pixels

					if frac_black_pixel > 0.6:
						# If more than 60% of the tile pixels are black, i.e. non-relevant part of the input map, continue with next tile but keep enumeration

						# skipped_tiles.append(padded_img_enumerator)
						skipped_tiles.append(img_enumerator)
						img_enumerator = img_enumerator + 1
						continue

				if save_tiles:

					output_file = os.path.join("Output_Tiles", image_name,"{}-{}-{}-{}-h_{}-{}-w".format(padded_img_enumerator,
																				 input_map, padded_h0, padded_h1,
																				 padded_w0, padded_w1))

					if input_map[0:4] == 'mask':
						# Save .csv
						np.savetxt(output_file + '.csv', tile, fmt='%i', delimiter=' ')
						# sample_mask_df.to_csv(output_file + '.csv',  header=False, index=False, sep=' ')

					else:
						# Save image
						Image.fromarray(tile).save(output_file + '.tif')

				img_enumerator = img_enumerator + 1

				# Append tiles
				# tiles = np.append(tiles, [tile], axis = 0)

		if input_map[0:3] == 'RGB':
			# Save skipped tile numbers
			np.savetxt(os.path.join("Output_Tiles", image_name) + '_skipped_tiles.txt', skipped_tiles, fmt = '%s')

		if split_train_test:
			# Split tiles into training and test tiles
			# left_index_of_test is the upper-left tile (enumerated according to img_enumerator) from where the splitting is performed
			split_train_test_dir(os.path.join('Output_Tiles', image_name), tiles_x, tiles_y, left_index_of_test)
			# split_train_test_dir(os.path.join('Output_Tiles', image_name), tiles_x, tiles_y, left_index_of_test, output_path = 'Output_Tiles')

	# return tiles


def add_skipped_tiles(prediction_array, kept_tiles_indices, tot_num_tiles):

	num_predictions, tile_pixel_height, tile_pixel_width, num_classes = prediction_array.shape

	# -1: Numpy indices start from 0, while the enumeration of tiles start from 1
	kept_tiles_indices = kept_tiles_indices - 1

	# Initialise the full array containing only blank (i.e. background) tiles
	full_array = np.zeros((tot_num_tiles, tile_pixel_height, tile_pixel_width, num_classes))

	# Write the prediction results into the tiles which were kept/not skipped in the tile_image step
	full_array[kept_tiles_indices] = prediction_array

	return full_array


def load_data(samples_directory, image_size, samples_type, repeat_DEM_tiles = 1, do_img_color_change=False):
	''' Load tiles (RGB, DEM, mask) into a single matrix.

	samples_directory: Path where tiles can be found
	image_size: Image size for the loaded tiles
	samples_type: Specification of the type of tile, i.e. 'RGB', 'DEM' or 'mask'
	repeat_DEM_tiles: If several RGB input years are used, the information from the DEM is just repeated for each year.
					  I.e. if input images from e.g. year 2000 and 2004 are used, set repeat_DEM_tiles = 2
	do_img_color_change: Flag whether color adjustments (image augmentation) shall be performed. Leads to additional samples
	'''

	filenames = os.listdir(samples_directory)
	filenames.sort()

	samples = []
	samples_return = []

	# Loop through each file, i.e. each sample
	for file in filenames:
		file_path = os.path.join(samples_directory, file)

		samples.append(file_path)

	# print(samples[0], '\n', samples[1], '\n', samples[2], '\n', samples[3], '\n')

	if samples_type == 'RGB':
		# samples_return = np.array([cv2.imread(file) for file in samples])
		samples_return = np.array(
			[cv2.resize(cv2.imread(file), dsize=image_size, interpolation=cv2.INTER_CUBIC).astype(int) for file in
			 samples])

		if do_img_color_change:
			# print(samples_return.shape)

			# Change colour of images
			samples_return_gray = np.array([cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY) for sample in samples_return])
			# samples_return_recoloured = np.array([cv2.applyColorMap(sample, cv2.COLORMAP_RAINBOW) for sample in samples_return])

			# Grayscale images have only one channel, i.e. it is not possible to append to 3-channel RGB images.
			# Repeat pixel values to form 3-channel image.
			samples_return_gray = np.repeat(samples_return_gray[:, :, :, np.newaxis], 3, axis=3)

			print(samples_return_gray.shape)
			# print(samples_return_recoloured.shape)

			samples_return = np.append(samples_return, samples_return_gray, axis=0)
		# samples_return = np.append(samples_return, samples_return_recoloured, axis = 0)



	elif samples_type == 'mask':
		# samples_return = np.array([np.loadtxt(file) for file in samples])
		samples_return = np.array([cv2.resize(np.loadtxt(file), dsize=image_size, interpolation=cv2.INTER_NEAREST).astype(int) for file in samples])

		if do_img_color_change:
			# If more images are added, the masks are just repeated for each patch of the added images.
			# Since recoloured images are appended to the end of the image array, do the same for the masks.
			# samples_return_temp = np.append(samples_return, samples_return, axis=0)
			# samples_return = np.append(samples_return_temp, samples_return, axis=0)

			samples_return = np.append(samples_return, samples_return, axis=0)

	elif samples_type == 'DEM':
		samples_return = np.array([gdalnumeric.LoadFile(file).astype(np.float32) for file in samples])

		# If four different years are considered, the DEM derivatives are just repeated for each patch of each year, i.e. repeat_DEN_tiles = 4.
		samples_return = np.repeat(samples_return, repeat_DEM_tiles, axis=0)

		if do_img_color_change:
			# If more images are added, the DEM derivatives are just repeated for each patch of the added images.
			# Since recoloured images are appended to the end of the image array, do the same for DEM derivatives.
			# samples_return_temp = np.append(samples_return, samples_return, axis = 0)
			# samples_return = np.append(samples_return_temp, samples_return, axis = 0)

			samples_return = np.append(samples_return, samples_return, axis=0)

	return samples_return


def plot_segmentation_results(images_test, mask_labels_test, net, timestamp, outcome_folder = None, threshold=0.5, samples_per_batch = 200, fontsize=40, figsize=(33, 33)):
	''' Obtain segmentation for (test) images and plot ground truth and prediction results for all classes jointly.

	images_test: Array of (test) images of shape (num_samples, image height, image width, channels)
	mask_labels_test: Array of one-hot encoded (test) labels of shape (num_samples, image height, image width, classes)
	net: U-Net instance
	timestamp: Identifying time stamp of trained U-Net
	outcome_folder: Folder to save plots to
	threshold: Probability threshold applied on pixel class probabilities for class assignment
	samples_per_batch: It might be required to divide the (test) images into smaller batches to do the predictions
	fontsize, figsize: Specifications of font and figure size for joint plot
	'''

	if not outcome_folder:
		outcome_folder = os.path.join('Output_Training','{}-Plots_outcome'.format(timestamp))

	if not os.path.exists(outcome_folder):
		os.makedirs(outcome_folder)

	# Identify path where the trained U-Net is saved (-> checkpoint)
	checkpoint = os.path.join("Output_Training", timestamp + "-unet_trained", "model.ckpt")

	num_samples = images_test.shape[0]

	# Identify number of classes (-1 to remove background class)
	num_classes = mask_labels_test.shape[-1] - 1

	# If margin_size_x is not specified (i.e. [] ), the correct margin sizes are calculated
	margin_size_x = np.array([])

	print('Plot segmentation results of the trained U-Net from', timestamp, 'for', num_samples, 'samples and threshold of', threshold, '.')

	images_test = normalise_images(images_test)

	num_batches = int(np.ceil(num_samples/samples_per_batch))

	# Initialise the (absolute) image enumeration at -1 s.t. the first image starts with 0
	sample_enum_images_test = -1

	class_cmaps = ['Reds_r', 'Oranges_r', 'Purples_r', 'Blues_r']
	class_names = ['Landslide', 'Livestock Trail', 'Management Effect', 'Sheet Erosion']

	for batch_number in range(num_batches):
		print("------- Predicting and plotting segmentations for images in the range of {} to {}, with a total of {} images.".format(batch_number*samples_per_batch, (batch_number+1)*samples_per_batch, num_samples))
		prediction = net.predict(checkpoint, images_test[batch_number*samples_per_batch:(batch_number+1)*samples_per_batch])

		# if margin_size_x.size == 0:
		if batch_number == 0:
			# Identify margin sizes in x and y (width and height) direction to crop the images_test accordingly
			margin_size_y, margin_size_x = ((np.array(images_test.shape[1:3]) - np.array(prediction.shape[1:3])) / 2).astype(int)
			print(images_test.shape, prediction.shape, margin_size_y, margin_size_x)

		for sample_enum_prediction in range(prediction.shape[0]):

			sample_enum_images_test = sample_enum_images_test + 1

			# Plot segmentation result for given sample / images_test
			fig, ax = plt.subplots(num_classes, 4, sharex=True, sharey=True, figsize=figsize)

			for row in range(num_classes):
				if num_classes == 1:
					if margin_size_x == 0:
						ax[0].imshow(images_test[sample_enum_images_test, :, :, 0], aspect="auto")
						ax[1].imshow(mask_labels_test[sample_enum_images_test, :, :, 1], cmap=class_cmaps[0], aspect="auto")
					else:
						ax[0].imshow(images_test[sample_enum_images_test, margin_size_y:-margin_size_y, margin_size_x:-margin_size_x, 0], aspect="auto")
						ax[1].imshow(mask_labels_test[sample_enum_images_test, margin_size_y:-margin_size_y, margin_size_x:-margin_size_x, 1], cmap=class_cmaps[0], aspect="auto")

					pred = prediction[sample_enum_prediction, :, :, 1]
					ax[2].imshow(pred, cmap=class_cmaps[0], aspect="auto")
					ax[3].imshow(pred > threshold, cmap=class_cmaps[0], aspect="auto")

					# Remove x and y axes ticks
					ax[0].set_xticks([])
					ax[1].set_xticks([])
					ax[2].set_xticks([])
					ax[3].set_xticks([])
					ax[0].set_yticks([])

					ax[0].set_ylabel(class_names[0], fontsize=fontsize)

					ax[0].set_title('Input', fontsize=fontsize)
					ax[1].set_title('Ground Truth', fontsize=fontsize)
					ax[2].set_title('Prediction', fontsize=fontsize)
					ax[3].set_title('Prediction > {}'.format(threshold), fontsize=fontsize)

				else:
					if margin_size_x == 0:
						ax[row, 0].imshow(images_test[sample_enum_images_test, :, :, 0], aspect="auto")
						ax[row, 1].imshow(mask_labels_test[sample_enum_images_test, :, :, row + 1], cmap=class_cmaps[row], aspect="auto")
					else:
						ax[row, 0].imshow(images_test[sample_enum_images_test, :, :, 0], aspect="auto")
						ax[row, 1].imshow(mask_labels_test[sample_enum_images_test, :, :, row + 1], cmap=class_cmaps[row], aspect="auto")

					pred = prediction[sample_enum_prediction, :, :, row+1]
					ax[row, 2].imshow(pred, cmap=class_cmaps[row], aspect="auto")
					ax[row, 3].imshow(pred > threshold, cmap=class_cmaps[row], aspect="auto")

					# Remove x and y axes ticks
					ax[row, 0].set_xticks([])
					ax[row, 1].set_xticks([])
					ax[row, 2].set_xticks([])
					ax[row, 3].set_xticks([])
					ax[row, 0].set_yticks([])

					ax[row, 0].set_ylabel(class_names[row], fontsize=fontsize)

					if row == 0:
						ax[0, 0].set_title('Input', fontsize=fontsize)
						ax[0, 1].set_title('Ground Truth', fontsize=fontsize)
						ax[0, 2].set_title('Prediction', fontsize=fontsize)
						ax[0, 3].set_title('Prediction > {}'.format(threshold), fontsize=fontsize)

			fig.tight_layout()

			fig.savefig(os.path.join("{}".format(outcome_folder), "prediction_test_image_{}.png".format(sample_enum_images_test)))


def map_segmentation(timestamp, net, original_image, tile_dir_RGB, tile_dir_aspect, tile_dir_curvature, tile_dir_slope, tiles_x, tiles_y, margin_size_x='', margin_size_y='',
					 image_size=(), num_classes=5, thresholds=[0.5], skipped_tiles = False, resolution_25cm=True, do_prediction=True, use_DEM=True, use_rest_class=False,
					 samples_per_batch = 100):
	''' Segmentation of a whole map with a trained U-Net. It is expected that the map was already divided into tiles. For each tile the segmentation is predicted and
	all prediction are combined to obtain the segmentation of the whole map.

	timestamp: Identifying time stamp of trained U-Net
	net: U-Net instance
	original_image: Path to the GeoTIFF of the original map
	tile_dir_RGB, tile_dir_aspect, tile_dir_curvature, tile_dir_slope: Paths to the directories containing tiles for RGB and DEM derivatives
	tiles_x: Number of tiles in x direction (horizontal)
	tiles_y: Number of tiles in y direction (vertical)
	margin_size_x: Pixel margin size which overlaps with other tiles in x direction (horizontal)
	margin_size_y: Pixel margin size which overlaps with other tiles in y direction (vertical))
	image_size: Image size for the loaded tiles
	num_classes: Number of classes including background
	threshold: Array of probability thresholds applied on pixel class probabilities for class assignment. This can be either just one value [0.5] or multiple values [0.1,0.2,...]
	skipped_tiles:
	resolution_25cm:
	do_prediction:
	samples_per_batch: It might be required to divide the images into smaller batches to do the predictions
	'''



	map_name = os.path.splitext(os.path.basename(tile_dir_RGB))[0]
	output_folder = map_name + "-shapes_" + timestamp


	if do_prediction:
		# Identify path where the trained U-Net is saved (-> checkpoint)
		checkpoint = os.path.join("Output_Training", timestamp + "-unet_trained", "model.ckpt")

		# If margin_size_x is not specified (i.e. [] ), the correct margin sizes are calculated
		margin_size_x = np.array([])

		# Retrieve filenames and sort them (otherwise the order of tiles is random)
		filenames_RGB = os.listdir(tile_dir_RGB)
		filenames_RGB.sort()

		if use_DEM:
			filenames_aspect = os.listdir(tile_dir_aspect)
			filenames_aspect.sort()
			filenames_curvature = os.listdir(tile_dir_curvature)
			filenames_curvature.sort()
			filenames_slope = os.listdir(tile_dir_slope)
			filenames_slope.sort()

		num_files = len(filenames_RGB)

		if skipped_tiles:
			# Identify the tile numbers which were not skipped in the tile_image call
			kept_tiles_indices = np.array([int(filenames_RGB[i].split('-')[0]) for i in range(len(filenames_RGB))])

		# Define U-Net architecture manually
		# net = unet.Unet(channels=6, n_class=num_classes, layers=3, features_root=32)

		for i in range(num_files):
			print("\n------- Load input for tile", i + 1, "of", num_files)

			file_RGB = filenames_RGB[i]
			input_map_RGB = os.path.join(tile_dir_RGB, file_RGB)

			if use_DEM:
				file_aspect = filenames_aspect[i]
				file_curvature = filenames_curvature[i]
				file_slope = filenames_slope[i]

				input_map_aspect = os.path.join(tile_dir_aspect, file_aspect)
				input_map_curvature = os.path.join(tile_dir_curvature, file_curvature)
				input_map_slope = os.path.join(tile_dir_slope, file_slope)

			image_RGB = cv2.imread(input_map_RGB)
			if len(image_size) > 0:
				image_RGB = cv2.resize(image_RGB, dsize=image_size, interpolation=cv2.INTER_CUBIC)
			image_RGB = image_RGB[np.newaxis, :, :, :]

			if use_DEM:
				image_aspect = gdalnumeric.LoadFile(input_map_aspect).astype(np.float32)
				if len(image_size) > 0:
					image_aspect = cv2.resize(image_aspect, dsize=image_size, interpolation=cv2.INTER_CUBIC)
				image_aspect = image_aspect[np.newaxis, :, :]
				image_curvature = gdalnumeric.LoadFile(input_map_curvature).astype(np.float32)
				if len(image_size) > 0:
					image_curvature = cv2.resize(image_curvature, dsize=image_size, interpolation=cv2.INTER_CUBIC)
				image_curvature = image_curvature[np.newaxis, :, :]
				image_slope = gdalnumeric.LoadFile(input_map_slope).astype(np.float32)
				if len(image_size) > 0:
					image_slope = cv2.resize(image_slope, dsize=image_size, interpolation=cv2.INTER_CUBIC)
				image_slope = image_slope[np.newaxis, :, :]

			if use_DEM:
				# Join RGB and DEM images to form an input image with six channels, i.e. of shape (1, width, height, 6)
				image = np.append(image_RGB, image_aspect[:, :, :, np.newaxis], axis=3)
				image = np.append(image, image_curvature[:, :, :, np.newaxis], axis=3)
				image = np.append(image, image_slope[:, :, :, np.newaxis], axis=3)
			else:
				image = image_RGB


			image = normalise_images(image)

			if i == 0:
				# Print information on shapes in first run
				if use_DEM:
					print("\nShapes of RGB, aspect, curvature and slope image", image_RGB.shape, image_aspect.shape,
						  image_curvature.shape, image_slope.shape)
				else:
					print("\nShapes of RGB", image_RGB.shape)


				# Prepare array to save images to
				images = np.empty(shape=(0,)+image.shape[1:])

			images = np.append(images, image, axis=0)

		num_images = images.shape[0]
		num_batches = int(np.ceil(num_images/samples_per_batch))

		for batch_number in range(num_batches):
			# Perform prediction batch-wise. This might be necessary if the total amount of images (and processing steps) exceeds the available memory on the GPU
			print("------- Predicting segmentations for images in the range of {} to {}, with a total of {} images.".format(batch_number * samples_per_batch, (batch_number + 1) * samples_per_batch, num_images))
			prediction_batch = net.predict(checkpoint, images[batch_number * samples_per_batch:(batch_number + 1) * samples_per_batch])

			if batch_number == 0:
				# Initialise empty prediction array
				prediction = np.empty((0,)+prediction_batch.shape[1:])

			prediction = np.append(prediction, prediction_batch, axis=0)

		# Identify margin sizes in x and y (width and height) direction to crop the test_image accordingly
		margin_size_y, margin_size_x = ((np.array(image.shape[1:3]) - np.array(prediction.shape[1:3])) / 2).astype(int)

		print("Shape of joint input, prediction, margin_size_y, margin_size_x:", images.shape, prediction.shape, margin_size_y, margin_size_x)

		# Shape of an individual tile
		tile_pixel_height, tile_pixel_width = prediction.shape[1:3]
		print("Shape of tile_pixel_height and _width:", tile_pixel_height, tile_pixel_width)

		# Class prediction for each individual pixel (without the probabilities to belong to the background class)
		prediction = prediction[:, :, :, 1:].round(decimals=4)

		if skipped_tiles:
			prediction = add_skipped_tiles(prediction_array=prediction, kept_tiles_indices=kept_tiles_indices, tot_num_tiles=tiles_x*tiles_y)

		# Arrange prediction tiles in original composition
		array = prediction.reshape(tiles_y, tiles_x, tile_pixel_height, tile_pixel_width, num_classes - 1)

		# Rearrange prediction tiles s.t. image can be reconstructed
		array = array.reshape((tiles_y, np.multiply(tiles_x, tile_pixel_height), tile_pixel_width, num_classes - 1),
							  order='F').reshape(
			(np.multiply(tiles_y, tile_pixel_height), np.multiply(tiles_x, tile_pixel_width), num_classes - 1))

		print("Shape of input array for shape file construction:", array.shape)

		if not os.path.isdir(os.path.join('Output_Segmentation', output_folder)):
			os.makedirs(os.path.join('Output_Segmentation', output_folder))

		# # Save prediction array for full image
		# for i in range(num_classes-1):
		# 	np.savetxt("Output_Segmentation/{0}/prediction_{1}_class-{2}.csv".format(output_folder, map_name, i+1), array[...,i], delimiter=' ', fmt = '%1.4f' )

		# Alternatively, pickle numpy array. This will allow faster loading of the numpy array, however requires more space
		with open(os.path.join("Output_Segmentation", output_folder, "prediction_{}.pickle".format(map_name)), 'wb') as f:
			pickle.dump(array, f, protocol=pickle.HIGHEST_PROTOCOL)

	else:
		array_path = os.path.join("Output_Segmentation", map_name + '-shapes_' + timestamp, "prediction_" + map_name)
		#
		# # Load prediction array for full image
		# for i in range(num_classes-1):
		# 	if i == 0:
		# 		array = np.loadtxt(array_path + '_class-{}.csv'.format(i+1), delimiter=' ')
		# 		array = array[..., np.newaxis]
		# 	else:
		# 		arr2append = np.loadtxt(array_path + '_class-{}.csv'.format(i+1), delimiter=' ')
		# 		array = np.append(array, arr2append[..., np.newaxis], axis=-1)

		# Alternatively, load pickled numpy array
		with open(array_path + '.pickle', 'rb') as f:
			array = pickle.load(f)

		print('Loaded array.shape', array.shape)


	# Which class indices are to be considered as identifying the background and the rest class?
	background_class_index = 0

	if use_rest_class:
		rest_class_index = num_classes
		print("\n------- Rest Class: background_class_index = {}, rest_class_index = {}.".format(background_class_index, rest_class_index))

		# Retrieve background class again. The background probabilities per pixel are given by
		# "1 - sum(other class probabilities)".
		bg_probs = 1 - array.sum(axis=-1)

		# Append background probabilities to the end of each entry in the array
		# array = np.append(array, bg_probs[..., np.newaxis], axis=-1)
		# background_class_index = num_classes
		# rest_class_index = 0
		# print("\n------- Background class appended: background_class_index = {}, rest_class_index = {}.".format(background_class_index, rest_class_index))

	# Identify class index per pixel with highest probability
	array_class_indices = np.argmax(array, axis=-1)
	array_class_probs = np.max(array, axis=-1)

	for threshold in thresholds:
		print("Map segmentation with threshold", threshold)

		# Threshold for class assignment
		array_thresholded_bool = (array_class_probs > threshold).astype(int)

		# array_thresholded contains "1" if the threshold is exceeded, otherwise "0".
		# Multiplying (entry-wise) class indices gives in each entry the correct class label if threshold exceeded ("1"),
		# otherwise class label "0" for the background is retained.
		# In the case that background probabilities are appended, class label "0" identifies the rest_class_index.
		# Note: +1 due to the fact that class indices start from 1, but numpy indices from 0
		array_thresholded = np.multiply(array_thresholded_bool, array_class_indices + 1)

		if use_rest_class:
			# Threshold background class
			bg_probs_thresholded_bool = (bg_probs > threshold).astype(int)

			# Where all class probabilities (including background) lay below the threshold (i.e. False <-> 0), set the rest_class_index for those pixels
			array_thresholded[(array_thresholded_bool.astype(bool) == 0) & (bg_probs_thresholded_bool == 0)] = int(rest_class_index)

			# Divide the rest class pixel into erosion-class-specific rest classes by using the erosion class with the largest probability
			array_thresholded[array_thresholded == rest_class_index] = (array_class_indices[array_thresholded == rest_class_index] + 1)*10

		file_path = os.path.join('Output_Segmentation', output_folder, "{}_{}_threshold-{}.shp".format(timestamp, map_name, threshold))

		# Open original image to retrieve information on geolocation and pixel resolution
		original_map = gdal.Open(original_image)

		x_upperleft, pixel_x_res, _, y_upperleft, _, pixel_y_res = original_map.GetGeoTransform()

		if not resolution_25cm:
			# Coordinates of upper-left pixel corner
			upperleft_corner = (x_upperleft + pixel_x_res * margin_size_x, y_upperleft + pixel_y_res * margin_size_y)

		else:
			# If the resolution is 25 cm instead of 50 cm, adjust array and margins
			upperleft_corner = (x_upperleft + 2 * pixel_x_res * margin_size_x, y_upperleft + 2 * pixel_y_res * margin_size_y)

			array_thresholded = np.repeat(array_thresholded, 2, axis=0)
			array_thresholded = np.repeat(array_thresholded, 2, axis=1)

		print("Shape of array_thresholded:", array_thresholded.shape)

		# Reference systems
		# 4326: WGS84
		# 2056: CH1903+ / LV95
		EPSG_reference_sys = 2056

		array2shape(file_name=file_path, array=array_thresholded, upperleft_corner=upperleft_corner, pixel_x_res=pixel_x_res, pixel_y_res=-pixel_y_res, background_class_index=background_class_index, EPSG_reference_sys=EPSG_reference_sys)