# Introduction
This repository is for extracting C3D or I3D features from videos. 

Please cite with the following BibTeX:

## Requirements
* Python 3
* CUDA
* numpy
* tqdm
* [PyTorch](http://pytorch.org/) (1.2)
* [torchvision](http://pytorch.org/)  
Recommend: the environment can be established by run

```
conda env create -f anomaly_feature.yaml
```

## Video preprocessing
1. Using Videos_to_frames_multiprocessing.py to generate frames from videos.

ps: if you want to extracted I3D feature, it is recommend that using denseflow (https://github.com/open-mmlab/denseflow) to generate RGB and Optical_flow images.

## Input list generation
1. Using the ./dataset/write_data_label_txt_new.py 

 generate the index file for running feature_extract.py. 

## Feature extraction
1. Using feature_extract.py to generate I3D or C3D features.

Using script 

```
python feature_extract.py --dataset shanghaitech --modelName i3d --snapshot ./model/i3d/i3d_model_weight/model_kinetics_rgb.pth --datamodal rgb
```

to extract i3d rgb features.

Using script 

```
python feature_extract.py --dataset shanghaitech --modelName i3d --snapshot ./model/i3d/i3d_model_weight/model_kinetics_flow.pth --datamodal flow
``` 

to extract i3d flow features.

Using script
 
 ```
 python feature_extract.py --dataset shanghaitech --modelName c3d --snapshot ./model/c3d/c3d.pickle --datamodal rgb
``` 
 
 to extract c3d features.

Using dataset_creater.py to generate final feature file.
