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

**Doc dependencies:**

```bash
pip install sphinx_rtd_theme
```

## Training

```
usage: main.py train [-h] [--images path] [--val_images path] [--backbone BACKBONE] [--head_type store] [--lr value] [--regularization_l2 value] [--momentum value] [--epochs value] [--debug] model path

positional arguments:
  model                 path to output model or checkpoint to resume from
  path                  path transform.py

optional arguments:
  -h, --help            show this help message and exit
  --images path         path to images
  --val_images path     path to images
  --backbone BACKBONE   backbone model
  --head_type store     head type
  --lr value            Learning rate
  --regularization_l2 value, -reg_l2 value
                        Weight regularization
  --momentum value      SGD Optimizer Momentum
  --epochs value        Epochs until end
  --debug               No shuffle

```

Example:

```bash
python -m colorization.main --verbose train /mnt/data/checkpoints/colorization/growing/UNet_bc64_d4.pth /mnt/data/checkpoints/colorization/growing/transform.py --images ~/datasets/Imagenet/test --val_images ~/datasets/Imagenet/val --backbone UNet_bc64_d4 --epochs 30
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
  --batch_size value  Epochs until end
  --img_limit value   Epochs until end
  --debug             Epochs until end
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