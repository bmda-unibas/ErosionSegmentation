# Semantic Segmentation of Soil Erosion Sites

This repository contains a basic implementation for segmenting soil erosion sites on aerial imagery 
with a U-Net. The code was developped in the context of our paper on "[**Identifying Soil Erosion Processes 
in Alpine Grasslands on Aerial Imagery with a U-Net Convolutional Neural Network**](https://www.mdpi.com/2072-4292/12/24/4149)" (*Remote Sensing*, 2020).

![U-Net for Soil Erosion Segmentation](https://www.mdpi.com/remotesensing/remotesensing-12-04149/article_deploy/html/images/remotesensing-12-04149-g005.png "U-Net for Soil Erosion Segmentation")
This image is available under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) (author: Maxim Samarin); please cite reference below. 


## Set up your environment 

Our implementation is written in Python (3.6) and uses TensorFlow (version 1.10). If you use the 
[Anaconda distribution](https://www.anaconda.com/) (recommended), you can set up your local environment by executing

```conda env create -f requirements.yml``` 

or, in case you use a GPU (recommended), 


```conda env create -f requirements_gpu.yml``` 

Note that the implementation makes use of some auxiliary R-scripts. To use the full 
capabilities of the provided implementation, a local R installation with the following 
R libraries is required:

- raster
- rgdal
- optparse

## Usage

An example for the usage will follow shortly!

## Reference

We build our implementation on [tf_unet](https://github.com/jakeret/tf_unet) and made some adjustments 
in the original code.

Are you interested to use some of our code for your research and investigations? In that case please 
cite our paper:

```
@article{SamarinZweifel2020,
  title={Identifying Soil Erosion Processes in Alpine Grasslands on Aerial Imagery with a U-Net Convolutional Neural Network},
  author={Samarin*, Maxim and Zweifel*, Lauren and Roth, Volker and Alewell, Christine},
  journal={Remote Sensing},
  volume={12},
  number={24},
  pages={4149},
  year={2020},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```

![Graphical Abstract](https://www.mdpi.com/remotesensing/remotesensing-12-04149/article_deploy/html/images/remotesensing-12-04149-ag.png "Graphical Abstract")