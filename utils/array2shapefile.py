# Author: Maxim Samarin (maxim.samarin@unibas.ch)
# Last Modification: 13.12.20
#
# Created with recipes from the python GDAL/OGR cookbook:
#	https://pcjericks.github.io/py-gdalogr-cookbook/raster_layers.html#create-raster-from-array
#	https://pcjericks.github.io/py-gdalogr-cookbook/raster_layers.html#raster-to-vector-line
# 	https://gis.stackexchange.com/questions/254444/deleting-selected-features-from-vector-ogr-in-gdal-python

from osgeo import gdal, ogr, osr
import os
import numpy as np 


def array2shape(file_name, array, upperleft_corner, pixel_x_res, pixel_y_res, background_class_index = 0, EPSG_reference_sys = 2056):

	#####################################################################################
	# array2raster part:

	cols = array.shape[1]
	rows = array.shape[0]
	upperleft_x = upperleft_corner[0]
	upperleft_y = upperleft_corner[1]
	geotransform = (upperleft_x, pixel_x_res, 0, upperleft_y, 0, -pixel_y_res)

	# Create raster layer (GTiff if an actual image shall be saved, otherwise create in memory)
	# raster_driver = gdal.GetDriverByName('GTiff')
	# raster_layer = raster_driver.Create('test.tif', cols, rows, 1, gdal.GDT_Byte)	
	raster_driver = gdal.GetDriverByName('Mem')
	raster_layer = raster_driver.Create('', cols, rows, 1, gdal.GDT_Byte)
	# raster_layer = raster_driver.Create(file_name, cols, rows, 1, gdal.GDT_Float32)

	# Set geotransform
	raster_layer.SetGeoTransform(geotransform)

	# Write array to raster_layer (first band)
	outband = raster_layer.GetRasterBand(1)
	outband.WriteArray(array)

	# Set spatial reference system
	raster_layerSRS = osr.SpatialReference()
	raster_layerSRS.ImportFromEPSG(EPSG_reference_sys)
	raster_layer.SetProjection(raster_layerSRS.ExportToWkt())

	# Get raster band for polygonize
	raster_layer_band = raster_layer.GetRasterBand(1)

	#####################################################################################
	# raster2shape part:

	# Create vector layer 
	shape_driver = ogr.GetDriverByName("ESRI Shapefile")

	if os.path.exists(file_name):
	    shape_driver.DeleteDataSource(file_name)

	outDataSource = shape_driver.CreateDataSource(file_name)
	vector_layer = outDataSource.CreateLayer(file_name, srs = raster_layerSRS, geom_type = ogr.wkbPolygon )

	# Create a field in the output layer for the class of the polygons 
	field_def = ogr.FieldDefn('Class', ogr.OFTInteger)
	vector_layer.CreateField(field_def)

	# Polygonise regions in raster_layer with same value. 
	# I.e. each region with same value becomes an individual object.
	gdal.Polygonize(raster_layer_band, raster_layer_band.GetMaskBand(), vector_layer, 0)


	# Select objects corresponding to background. For background_class_index = 0 this would be Class = 0
	vector_layer.SetAttributeFilter("Class = {}".format(background_class_index))

	# Delete objects from vector layer
	for feat in vector_layer:
		vector_layer.DeleteFeature(feat.GetFID())


	# Delete drivers and datasources
	outDataSource.Destroy()
	raster_layer = None
	vector_layer = None




def test_conversion():
	'''
	Small test example.
	'''

	upperleft_corner = (-123.25745,45.43013)
	pixel_x_res = 10
	pixel_y_res = 10
	EPSG_reference_sys = 2056
	file_name = 'test.shp'
	background_class_index = 1

	array = np.array([[ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
		[ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
		[ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1],
		[ 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
		[ 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1],
		[ 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
		[ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
		[ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
		[ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2],
		[ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2],
		[ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2]])

	raster_layer = array2shape(file_name=file_name, array=array, upperleft_corner=upperleft_corner, pixel_x_res=pixel_x_res, pixel_y_res=pixel_y_res, background_class_index=background_class_index, EPSG_reference_sys=EPSG_reference_sys)



if __name__ == "__main__":

	# Test: Uncomment the following line and comment out all subsequent lines
	test_conversion()

	# # Specify file with prediction array
	# input_name = 'prediction_array.csv'
	#
	# file_name = os.path.splitext(os.path.basename(input_name))[0] + '.shp'
	#
	# array = np.loadtxt(input_name)
	#
	# background_class_index = 1
	#
	# # Specify coordinates of the upper-left corner and pixel resolution (in x/y direction)
	# upperleft_corner = (2681809.9086834, 1162590.21504)
	# pixel_x_res = 0.5
	# pixel_y_res = 0.5
	#
	# # Reference systems
	# # 4326: WGS84
	# # 2056: CH1903+ / LV95
	# EPSG_reference_sys = 2056
	#
	# raster_layer = array2shape(file_name=file_name, array=array, upperleft_corner=upperleft_corner, pixel_x_res=pixel_x_res, pixel_y_res=pixel_y_res, background_class_index=background_class_index, EPSG_reference_sys=EPSG_reference_sys)