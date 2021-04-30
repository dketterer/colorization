# Colorization

In this project I colorize black-and-white images with the help of neural networks. 
I started it as a research project for the University Esslingen but try to further maintain and develop it.

It started in 2019 with a U-Net and L1/L2 loss. The code from then is written in Tensorflow 1/Keras. 
You still find that code on the master branch.

This branch is a rewrite in Pytorch.

## Prerequisites

**Development dependencies:**

```
conda install pytorch==1.8.1 torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
conda install -c conda-forge opencv jupyterlab albumentations matplotlib torchvision
conda install tqdm protobuf

```

I work with a self compiled Pytorch to get the latest Cudnn support, 
but it should not make a difference if you use the same Pytorch version from the official release channels. 
The currently included CuDNN 8.0.0 has poor support for Ampere GPUs.

Install **CUDA_SLIC**:

```
PATH=/usr/local/cuda-11/bin:$PATH CUDA_INC_DIR=/usr/local/cuda-11/include pip install cuda-slic
```


**Doc dependencies:**

```bash
pip install sphinx_rtd_theme
```

## Data preparation

I use the ImageNet dataset.


## Training

```
usage: main.py train [-h] [--images path] [--val_images path] [--backbone BACKBONE] [--head_type HEAD_TYPE] [--lr value] [--regularization_l2 value] [--momentum value] [--optimizer selection] --iterations value
                     [--val_iterations value] [--warmup iterations] [--milestones [iterations [iterations ...]]] --growing_parameters path [--debug]
                     path path

positional arguments:
  path                  path to output model or checkpoint to resume from
  path                  path transform.py

optional arguments:
  -h, --help            show this help message and exit
  --images path         path to images
  --val_images path     path to images
  --backbone BACKBONE   backbone model
  --head_type HEAD_TYPE
                        head type
  --lr value            Learning rate
  --regularization_l2 value, -reg_l2 value
                        Weight regularization
  --momentum value      SGD Optimizer Momentum
  --optimizer selection
                        The Optimizer
  --iterations value, -iters value
                        How many mini batches
  --val_iterations value, -val_iters value
                        Validation run after how many mini batches
  --warmup iterations   numer of warmup iterations
  --milestones [iterations [iterations ...]]
                        List of iteration indices where learning rate decays
  --growing_parameters path
                        Json file with the params fpr batch size and image size
  --debug               No shuffle
```

Example:

```bash
python -m colorization.main --verbose train Resnext50_UNet.pth \
    transform.py \
    --backbone Resnext50_UNet \
    --images /mnt/data/datasets/ILSVRC2017_CLS-LOC/ILSVRC/Data/CLS-LOC/train \
    --val_images /mnt/data/datasets/ILSVRC2017_CLS-LOC/ILSVRC/Data/CLS-LOC/val \
    --lr 0.0003 \
    --growing_parameters growing_parameters.json \
    --iterations 70000 \
    --milestones 30000 55000 \
    --warmup 3000 \
    --val_iterations 3500
    
```

Examples for [growing_parameters.json](resources/growing_parameters.json) and 
[transform.py](resources/transform.py) are in the resources folder.

Growing parameters lets you define a batch size and image size starting from an iteration:  
Elements are: `"iteration": [batch_size, width, height]`.
```
{
	"0": [128, 128, 128],
	"10000": [64, 256, 256]
}
```



## Inference / Testing

```
usage: main.py infer [-h] [--images path] [--target_path path] [--batch_size value] [--img_limit value] [--debug] model

positional arguments:
  model               path to output model or checkpoint to resume from

optional arguments:
  -h, --help          show this help message and exit
  --images path       path to images
  --target_path path  path to images
  --batch_size value  Batch size
  --img_limit value   Only first N images in the folder
  --debug             Stich original image, grey and predicted together
```
## Generate Color Segmentation Masks

### Linear colour segmentation revisited

From the paper [Linear colour segmentation revisited, Smagina et al., 2019, SPIE](https://spie.org/Publications/Proceedings/Paper/10.1117/12.2523007?SSO=1). 

The segmentation masks I use are rather large, about 20 areas per image.

I [forked](https://github.com/dketterer/colorsegmentation) their repo and fixed some issues to make it compatible with OpenCV 4.2.

See [utils/create_color_segments_multi.py](utils/create_color_segements_multi.py) for the usage.

```
python3 utils/create_color_segements_multi.py
```


## Unittest

Run the unittests from the top level directory:

```bash
python -m unittest
```

## Build the docs

```bash
cd docs
make html
```