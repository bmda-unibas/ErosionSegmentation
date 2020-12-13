# Author: Maxim Samarin (maxim.samarin@unibas.ch)
# Last modification: 13.12.20
#

import os
import time
import psutil

import numpy as np


from tf_unet import unet
from tf_unet.image_util import SimpleDataProvider

from utils.utils import load_data, plot_segmentation_results, map_segmentation



if __name__ == '__main__':
    start_time = time.strftime('%m-%d-%Y_%H%M')
    print(time.strftime('Starting at: %H:%M:%S, %d.%m.%y\n'))

    only_test_all_samples = False
    continue_training = False

    # Specify image size as (width, height), i.e. (pixels in y direction, pixels in x direction)
    # All inputs are rescaled to ensure same input size
    image_size = (388, 352)

    if only_test_all_samples:
        # Set the time stamp of the considered U-Net run manually
        timestamp = '08-26-2019_0908'
    elif continue_training:
        # Set the time stamp of the U-Net for which the training shall be continued
        timestamp = '08-26-2019_0908'

        print("Continue training model from U-Net run", timestamp)
    else:
        timestamp = start_time


    data_dir = 'Input_Tiles/'

    RGB_samples_dir_train = data_dir + 'Training/images'
    RGB_samples_dir_test = data_dir + 'Testing/images'

    mask_samples_dir_train = data_dir + 'Training/label_mask'
    mask_samples_dir_test = data_dir + 'Testing/label_mask'

    DEM_samples_dir_aspect_train = data_dir + 'aspect'
    DEM_samples_dir_aspect_test = data_dir + 'aspect'
    DEM_samples_dir_curvature_train = data_dir + 'curv'
    DEM_samples_dir_curvature_test = data_dir + 'curv'
    DEM_samples_dir_slope_train = data_dir + 'slope'
    DEM_samples_dir_slope_test = data_dir + 'slope'

    # If several RGB input years are used, the information from the DEM is just repeated for each year.
    # I.e. if input images from e.g. year 2000 and 2004 are used, set repeat_DEM_tiles = 2. If only 2000 is used -> repeat_DEM_tiles = 1
    repeat_DEM_tiles = 1

    #####################################################################################################################################
    # Read-in data:

    images_train = load_data(samples_directory = RGB_samples_dir_train, image_size = image_size, samples_type='RGB')

    print("Shape of images for training", images_train.shape)

    images_test = load_data(samples_directory = RGB_samples_dir_test, image_size = image_size, samples_type='RGB')

    print("Shape of images for testing", images_test.shape)

    masks_train = load_data(samples_directory = mask_samples_dir_train, image_size = image_size, samples_type='mask')

    unique_classes_train = np.unique(masks_train)
    print("Shape of masks for training", masks_train.shape, "with unique class labels:", unique_classes_train)

    masks_test = load_data(samples_directory = mask_samples_dir_test, image_size = image_size, samples_type='mask')

    unique_classes_test = np.unique(masks_test)
    print("Shape of masks for testing", masks_test.shape, "with unique class labels:", unique_classes_test)

    DEM_aspect_train = load_data(samples_directory = DEM_samples_dir_aspect_train, image_size = image_size, samples_type='DEM', repeat_DEM_tiles = repeat_DEM_tiles)

    print("Shape of aspect images for training", DEM_aspect_train.shape)
    images_train = np.append(images_train, DEM_aspect_train[:,:,:,np.newaxis], axis = 3)

    DEM_aspect_test = load_data(samples_directory = DEM_samples_dir_aspect_test, image_size = image_size, samples_type='DEM')

    print("Shape of aspect images for testing", DEM_aspect_test.shape)
    images_test = np.append(images_test, DEM_aspect_test[:,:,:,np.newaxis], axis = 3)

    DEM_curvature_train = load_data(samples_directory = DEM_samples_dir_curvature_train, image_size = image_size, samples_type='DEM')

    print("Shape of curvature images for training", DEM_curvature_train.shape)
    images_train = np.append(images_train, DEM_curvature_train[:,:,:,np.newaxis], axis = 3)

    DEM_curvature_test = load_data(samples_directory = DEM_samples_dir_curvature_test, image_size = image_size, samples_type='DEM')

    print("Shape of curvature images for testing", DEM_curvature_test.shape)
    images_test = np.append(images_test, DEM_curvature_test[:,:,:,np.newaxis], axis = 3)

    DEM_slope_train = load_data(samples_directory = DEM_samples_dir_slope_train, image_size = image_size, samples_type='DEM')

    print("Shape of slope images for training", DEM_slope_train.shape)
    images_train = np.append(images_train, DEM_slope_train[:,:,:,np.newaxis], axis = 3)

    DEM_slope_test = load_data(samples_directory = DEM_samples_dir_slope_test, image_size = image_size, samples_type='DEM')

    print("Shape of slope images for testing", DEM_slope_test.shape)
    images_test = np.append(images_test, DEM_slope_test[:,:,:,np.newaxis], axis = 3)

    num_classes = len(unique_classes_train)

    # Transform labels to one-hot encoding
    mask_labels_train = np.eye(num_classes)[masks_train.astype(int)].astype(int)
    mask_labels_test = np.eye(num_classes)[masks_test.astype(int)].astype(int)

    print("Training: Images_train shape ", images_train.shape, "Mask_labels_train shape", mask_labels_train.shape)
    print("Test: Images_test shape ", images_test.shape, "Mask_labels_test shape", mask_labels_test.shape)

    memory_rss = psutil.Process(os.getpid()).memory_info().rss
    print("\nMemory info: Physical memory used (rss) =", memory_rss/(10**9), "GB.\n")

    #####################################################################################################################################

    if only_test_all_samples:
        #
        images_test = np.append(images_train, images_test, axis=0)
        mask_labels_test = np.append(mask_labels_train, mask_labels_test, axis=0)

        print("Only testing! Merge training and test sets")
        print("Images shape ", images_test.shape, "Mask_labels shape", mask_labels_test.shape)

        data_provider = SimpleDataProvider(data=images_test, label=mask_labels_test)
        net = unet.Unet(channels=data_provider.channels, n_class=data_provider.n_class, layers=3, features_root=32)

    else:

        data_provider = SimpleDataProvider(data = images_train, label = mask_labels_train)

        net = unet.Unet(channels=data_provider.channels, n_class=data_provider.n_class, layers=3, features_root=32)

        batch_size = 20

        trainer = unet.Trainer(net, batch_size=20, verification_batch_size = 10, optimizer="adam")

        if continue_training:
            print("Restore model of U-Net run", timestamp)

            path = trainer.train(data_provider, output_path="Output_Training/{}-unet_trained".format(timestamp), training_iters=images_train.shape[0]//batch_size,
                                 epochs=10, dropout=0.9, display_step=20, restore=True, prediction_path="Output_Training/{}-prediction".format(timestamp))

        else:
            path = trainer.train(data_provider, output_path="Output_Training/{}-unet_trained".format(start_time), training_iters=images_train.shape[0]//batch_size,
            epochs=100, dropout=0.9, display_step=20, prediction_path="Output_Training/{}-prediction".format(start_time))

        print('Started at:', start_time, '\n')
        print(time.strftime('\nFinished training at: %H:%M:%S, %d.%m.%y'))


    plot_segmentation_results(images_test=images_test, mask_labels_test=mask_labels_test, net=net, timestamp=timestamp, threshold=0.5)

    print(time.strftime('\nFinished plotting at: %H:%M:%S, %d.%m.%y'))

    #####################################################################################################################################
    # Segmentation for a completely new input

    original_image = 'Input/full_input_image.gif'

    tile_dir_RGB = 'Input_Tiles/Validation'

    tile_dir_aspect = 'Input_Tiles/aspect'
    tile_dir_curvature = 'Input_Tiles/curvature'
    tile_dir_slope = 'Input_Tiles/slope'

    tiles_x = 25
    tiles_y = 19


    threshold = [0.3]

    map_segmentation(timestamp=timestamp, net=net, original_image=original_image, tile_dir_RGB=tile_dir_RGB, tile_dir_aspect=tile_dir_aspect,
                     tile_dir_curvature=tile_dir_curvature, tile_dir_slope=tile_dir_slope, tiles_x=tiles_x, tiles_y=tiles_y, image_size=image_size,
                     num_classes=num_classes, thresholds=threshold)


    print(time.strftime('\nFinished map_segmentation at: %H:%M:%S, %d.%m.%y'))