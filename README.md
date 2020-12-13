# Segmentation of Soil Erosion

This repository contains a basic implementation for segmenting soil erosion sites on aerial imagery 
with a U-Net. The code was developped in the context of our paper on "**Identifying Soil Erosion Processes 
in Alpine Grasslands on Aerial Imagery with a U-Net Convolutional Neural Network**" (currently under review 
in Remote Sensing).



## Setup your environment 

Our implementation is written in Python (3.6) and uses TensorFlow (V1.10). If you use the 
[Anaconda distribution](https://www.anaconda.com/) (recommended), you can set up your local environment by executing

```conda env create -f requirements.yml``` 

or, for execution on a GPU (recommended), 


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
  author={Samarin*, Maxim and Zweifel, Lauren* and Roth, Volker and Alewell, Christine},
  journal={Under review in Remote Sensing},
  year={2020},
}
```