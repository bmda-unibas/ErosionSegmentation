# Author: Maxim Samarin (maxim.samarin@unibas.ch)
# Date: 13.12.20
#
#

library(raster)
library(rgdal)
library(optparse)

# Parse arguments from command line 
option_list = list(
  make_option(c("-o", "--shapes_OBIA"), type="character", default=NULL, help="OBIA segmentation shape file", metavar="character"),  
  make_option(c("-n", "--shapes_UNet"), type="character", default=NULL, help="UNet segmentation shape file", metavar="character"),  
  make_option(c("-p", "--map_size"), type="character", default=NULL, help="Size of the raster map", metavar="character")
); 

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

num_classes <-  4
# 1 <- 'Bare Soil Landslide', 2 <- 'Bare Soil Live-Stock Trails', 3 <- 'Bare Soil Fields', 4 <-  'Soil Erosion'
class_names <- c('Bare Soil Landslide','Bare Soil Live-Stock Trails','Bare Soil Fields','Soil Erosion')

# In the OBIA results the mapping between class indices is slightly different:
# 1 <- 'Bare Soil Fields', 2 <- 'Bare Soil Landslide', 3 <- 'Bare Soil Live-Stock Trails', 4 <-  'Soil Erosion'
obia_class_indices <- seq(1,num_classes)
names(obia_class_indices) <- c('Bare Soil Fields','Bare Soil Landslide','Bare Soil Live-Stock Trails','Soil Erosion')


# Convert map_size from string to numeric, s.t. opt$map_size[1] = map_size in y/height direction and opt$map_size[2] = map_size in x/width direction
# opt$map_size <- '5968,8740'
opt$map_size <- as.numeric(strsplit(opt$map_size,',')[[1]])

polygons_OBIA <- readOGR(opt$shapes_OBIA)
polygons_UNET <- readOGR(opt$shapes_UNet)

# Dummy raster with as many rows and columns as in UNet prediction matrix specified by map_size
r <- raster(ncol=opt$map_size[2], nrow=opt$map_size[1])

# Assign extent of UNET results to raster
extent(r) <- extent(polygons_UNET)

# Check coordinate reference systems
# crs(polygons_OBIA)
# crs(polygons_UNET)

# Clip OBIA results to raster/UNET extent
polygons_OBIA@bbox <- as.matrix(extent(r))

# Check coordinate reference systems and extent
# crs(polygons_OBIA)
# crs(polygons_UNET)
# extent(polygons_OBIA)
# extent(polygons_UNET)
# extent(r)

for (class_index in seq(num_classes)) {
  polygons_OBIA_selection <- polygons_OBIA[polygons_OBIA$Class == class_names[class_index], ]
  polygons_UNET_selection <- polygons_UNET[polygons_UNET$Class == class_index, ]
  
  # Transform vector layers to rasters.
  # In OBIA results, class = 2 correpsonds to LS
  # in UNet results, class = 1 correpsonds to LS.
  raster_OBIA_selection <- rasterize(polygons_OBIA_selection, r, 'Class')
  raster_UNET_selection <- rasterize(polygons_UNET_selection, r, 'Class')
  
  # plot(raster_OBIA_selection)
  # plot(raster_UNET_selection)
  
  # Where UNET results overlap with OBIA results, save UNET class index,
  # no overlap of UNET with OBIA results -> 0, all other NA.
  raster_masked <- mask(raster_UNET_selection, raster_OBIA_selection, updatevalue = 0)
  
  # Alernatively: Where OBIA results overlap with UNET results, save OBIA class index,
  # no overlap of OBIA with UNET results -> 0, all other NA.
  # raster_masked <- mask(raster_OBIA_selection, raster_UNET_selection, updatevalue = 0)
  
  raster_masked.matrix <- as.matrix(raster_masked, dimnames = NULL)
  
  # plot(raster_masked)
  
  # Optionally: Change indices: 1 <-> overlap of UNet results with OBIA results
  # raster_masked.matrix[raster_masked.matrix == class_index] <- 1
  # raster_masked.matrix[raster_masked.matrix == obia_class_indices[[class_names[class_index]]]] <- 1
  
  # plot(raster(raster_masked.matrix), col = c("red", "blue"))
  # title('OBIA-UNET comparison; Blue: Match, Red: Mismatch')
  
  # unique(c(raster_masked.matrix))
  
  cat("\n\n-------Results for", class_names[class_index], ":\n")
  
  # Convert matrix to table yielding relative frequency
  raster_masked.table <- table(raster_masked.matrix, dnn = paste('No overlap of U-Net with OBIA pixels -> 0; overlap of U-Net with OBIA pixels ->', class_index))
  print(raster_masked.table)
  print(raster_masked.table/sum(raster_masked.table))
  }