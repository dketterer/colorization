## Install

```
conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
conda install -c conda-forge opencv jupyterlab albumentations matplotlib torchvision
conda install tqdm

```


## Unittest

Run the unittests:

```bash
python -m unittest
```

## Train

```bash
python -m colorization.main --verbose train /mnt/data/checkpoints/colorization/growing/UNet_bc64_d4.pth /mnt/data/checkpoints/colorization/growing/transform.py --images ~/datasets/Imagenet/test --val_images ~/datasets/Imagenet/val --backbone UNet_bc64_d4 --epochs 30
```