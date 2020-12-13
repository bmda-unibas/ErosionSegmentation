# Author: Maxim Samarin
# Last modification: 13.12.20
#
# With this script, the labelled/annotated regions in an aerial image, e.g. landslides, can be extracted.
# This means, that the aerial image (input_map) is converted to a matrix, where each entry corresponds to a pixel of the aerial image.
# Pixels, which overlap with the polygons of the annotated class are flagged with integers 1,2,3,...; for example landslide pixels are flagged with value 1.
# Otherwise, non-overlapping pixels are flagged with value 0. 
# Some input_map might have a larger extent the raster spans, i.e. there are empty (NA) pixels. NA pixels are set to 0 (background).
# This script returns the mask matrix as a .csv.  
#
# Run with: Rscript --vanilla extract_landslides.R --input_map "INPUT_MAP_PATH" --polygons "POLYGONS_PATH" --output "OUTPUT_FOLDER" --output_name "FILE_NAME"
# Same as: Rscript --vanilla extract_landslides.R -i "INPUT_MAP_PATH" -p "POLYGONS_PATH" -o "OUTPUT_FOLDER" -n "FILE_NAME"


# Import libraries. If a library is missing, try for each missing library the following line
#install.packages("raster", dependencies = TRUE)
library(raster)
library(rgdal)
library(optparse)


# Parse arguments from command line 
option_list = list(
  make_option(c("-i", "--input_map"), type="character", default=NULL, help="Input map", metavar="character"),
  make_option(c("-p", "--polygons"), type="character", default=NULL, help="Polygon mask", metavar="character"),  
  make_option(c("-o", "--output"), type="character", default="Output", help="Output folder [default= %default]", metavar="character"),
  make_option(c("-n", "--output_name"), type="character", default="matrix", help="Output folder [default= %default]", metavar="character")
); 

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

# Set a working directory, if necessary
#setwd("/home/...")

# Load raster layer by specifing the input_map explicitly. With raster(...) only the first band (R) is retrieved, otherwise use brick(...)
#?brick
rl <- raster(opt$input_map)
#rl
# projection(rl)

# CRS projection specification. With swisstopo tiles, use projection_swisstopo
projection_swisstopo <- '+proj=somerc +lat_0=46.95240555555557 +lon_0=7.439583333333334 +k_0=1 +x_0=600000 +y_0=200000 +ellps=bessel +towgs84=674.4,15.1,405.3,0,0,0,0 +units=m +no_defs'
# projection_LV95 <- '+proj=somerc +lat_0=46.95240555555556 +lon_0=7.439583333333333 +k_0=1 +x_0=2600000 +y_0=1200000 +ellps=bessel +towgs84=674.374,15.056,405.346,0,0,0,0 +units=m +no_defs'
projection_LV95 <- "+proj=somerc +lat_0=46.95240555555556 +lon_0=7.439583333333333 +k_0=1 +x_0=2600000 +y_0=1200000 +ellps=bessel +units=m +no_defs"

# Change crs projection of raster layer
if (is.na(projection(rl))) {
  crs(rl) <- projection_LV95
} else {
  rl <- projectRaster(rl, crs=projection_LV95)
}
# projection(rl)

# If the raster layer has margins (columns, rows with only NA), trim margins
#trim(rl)

# Get matrix of raster layer, each entry is the respective R-value of correspondig pixel 
#rl.matrix <- as.matrix(rl)


# Plot raster layer (note that only R-band is considered)
# plot(rl)

# Load polygon (vector) layer explicitly
polygons <- readOGR(opt$polygons)

cat("\n! Raster layer and shape layer have been loaded.\n")

#summary(polygons)

# Change projection of vector layer
if (is.na(projection(rl))) {
  polygons <- spTransform(polygons, CRS(projection_swisstopo))
  # polygons_bb <- spTransform(polygons_bb, CRS(projection_swisstopo))
  # polygons_geb <- spTransform(polygons_geb, CRS(projection_swisstopo))
} else {
  if (!(projection(polygons) == projection(rl))) {
    polygons <- spTransform(polygons, CRS(projection(rl)))
  }
  # if (!(projection(polygons_bb) == projection(rl))) {
  # polygons_bb <- spTransform(polygons_bb, CRS(projection(rl)))
  # }
  # if (!(projection(polygons_geb) == projection(rl))) {
  # polygons_geb <- spTransform(polygons_geb, CRS(projection(rl)))
  # }
}

# Each object, i.e. landslide, has its own id
# polygons$OBJECTID

# Ensure that vector layer has the same extent as raster layer
polygons@bbox <- as.matrix(extent(rl))
# polygons_bb@bbox <- as.matrix(extent(rl))
# polygons_geb@bbox <- as.matrix(extent(rl))

# extent(rl)

# If polygons contains classes such as "livestock trails" and "management degredation", save separately
polygons_LS <- polygons[polygons$Class == "Bare Soil Landslide", ]
polygons_live <- polygons[polygons$Class == "Bare Soil Live-Stock Trails", ]
polygons_field <- polygons[polygons$Class == "Bare Soil Fields", ]
polygons_soil_erosion <- polygons[polygons$Class == "Soil Erosion", ]


cat("\n! Setting mask classes.\n")

# Extract patches where raster and polygon layers overlap, for non-overlapping pixels, set value 0
#?mask
# Encoding: BG -> 0, LS -> 1, live-stock trail -> 2, field -> 3, soil erosion -> 4
rl_masked_LS <- mask(rl, polygons_LS, updatevalue = 0)
rl_masked_LS.matrix <- as.matrix(rl_masked_LS, dimnames = NULL)
rl_masked_LS.matrix[rl_masked_LS.matrix != 0] <- 1

rl_masked_live <- mask(rl, polygons_live, updatevalue = 0)
rl_masked_live.matrix <- as.matrix(rl_masked_live, dimnames = NULL)
rl_masked_live.matrix[rl_masked_live.matrix != 0] <- 2

rl_masked_field <- mask(rl, polygons_field, updatevalue = 0)
rl_masked_field.matrix <- as.matrix(rl_masked_field, dimnames = NULL)
rl_masked_field.matrix[rl_masked_field.matrix != 0] <- 3

rl_masked_soil_erosion <- mask(rl, polygons_soil_erosion, updatevalue = 0)
rl_masked_soil_erosion.matrix <- as.matrix(rl_masked_soil_erosion, dimnames = NULL)
rl_masked_soil_erosion.matrix[rl_masked_soil_erosion.matrix != 0] <- 4


# Summing up matrices: Background pixels are identified with 0, adding class label pixelwise. 
# rl_masked.matrix <- rl_masked_LS.matrix + rl_masked_live.matrix + rl_masked_field.matrix + rl_masked_soil_erosion.matrix

# In some cases (two classes are directly adjacent), pixels might have two class labels. Though labels are 0,1,2,3,4 another class label 5 was observed.
# Workaround: Do not add rl_masked_LS.matrix, but set all pixels with a "1" in rl_masked_LS.matrix to "1" in rl_masked.matrix. 
# The reason for this to work is: - "5" occurs when "1+4" happens, so first set all "4"s
#                                 - "5" might also occur due to "2+3". But if we do not observe "5"s with the workaround, they are only due to "1+4"
#                                 - All other cases ("6","7") do not occur and the workaround also prevents spurious "1+2=3" and "1+3=4" events
# rl_masked.matrix <- rl_masked_live.matrix + rl_masked_field.matrix + rl_masked_soil_erosion.matrix
# rl_masked.matrix <- replace(rl_masked.matrix, rl_masked_LS.matrix != 0, 1)

# # MWE replace: 
# mat_mwe = matrix(c(0,0,2,3,4,0,0,0,0,4,4,4,0,0,0), ncol = 5, nrow = 3)
# mat_mwe_mask = matrix(c(0,0,0,0,0,0,0,0,0,0,1,1,0,0,0), ncol = 5, nrow = 3)
# mat_result = replace(mat_mwe, mat_mwe_mask != 0, 1)
# mat_mwe
# mat_result

rl_masked.matrix <- rl_masked_field.matrix
rl_masked.matrix <- replace(rl_masked.matrix, rl_masked_soil_erosion.matrix != 0, 4)
rl_masked.matrix <- replace(rl_masked.matrix, rl_masked_live.matrix != 0, 2)
rl_masked.matrix <- replace(rl_masked.matrix, rl_masked_LS.matrix != 0, 1)

# Replace NA values with 0 (i.e. set to background)
rl_masked.matrix[is.na(rl_masked.matrix)] <- 0

# Check which class labels are contained in matrix
cat("\n! Unique class indices.\n")
unique(c(rl_masked.matrix))

#rl_masked.matrix <- replace(rl_masked.matrix, rl_masked.matrix != 0, 1)
#rl_masked.matrix

# Structure of rl_masked
#str(rl_masked)
#str(rl_masked.matrix)

# How many NaNs are there?
#nan_num <- sum(sum(is.na(rl_masked.matrix)))

#dim_mat <- dim(rl_masked.matrix)

# How many cells, i.e. pixels, are not NaN?
#prod(dim_mat) - nan_num

# Plot extracted patches
#plot(rl_masked_LS)
# plot(rl_masked)
# plot(rl_masked.matrix)

cat("\n! Saving mask to csv.\n")

# Save matrix as csv
if (!dir.exists(opt$output)){dir.create(opt$output)}
out_name <- paste("erosion_classes_", opt$output_name, ".csv", sep="")
write.table(rl_masked.matrix, file = paste(opt$output, out_name, sep = "/"), row.names = FALSE, col.names = FALSE)

