# Author: Maxim Samarin	(maxim.samarin@unibas.ch)
# Last modification: 13.12.20
#

from tf_unet import unet

from utils.utils import map_segmentation

def urs2016(timestamp, num_classes, thresholds, net):
    original_image = 'Input/full_input_image.gif'
    tile_dir_RGB = 'Input_Tiles/New_Example_RGB_tiles'
    tile_dir_aspect = 'Input_Tiles/New_Example_Aspect_tiles'
    tile_dir_curvature = 'Input_Tiles/New_Example_Curvature_tiles'
    tile_dir_slope = 'Input_Tiles/New_Example_Slope_tiles'

    tiles_x = 46
    tiles_y = 39

    margin_size_x = 20
    margin_size_y = 20

    image_size = (388, 352)

    resolution_25cm = True
    do_prediction = True

    map_segmentation(timestamp=timestamp, net=net, original_image=original_image, tile_dir_RGB=tile_dir_RGB,
                     tile_dir_aspect=tile_dir_aspect,
                     tile_dir_curvature=tile_dir_curvature, tile_dir_slope=tile_dir_slope, tiles_x=tiles_x,
                     tiles_y=tiles_y,
                     margin_size_x=margin_size_x, margin_size_y=margin_size_y, image_size=image_size,
                     num_classes=num_classes,
                     thresholds=thresholds, resolution_25cm=resolution_25cm,
                     do_prediction=do_prediction)


if __name__ == '__main__':

    timestamp = '08-26-2019_0908'

    num_classes = 5

    thresholds = [0.3]

    net = unet.Unet(channels=6, n_class=num_classes, layers=3, features_root=32)

    urs2016(timestamp=timestamp, num_classes=num_classes, thresholds=thresholds, net=net)
