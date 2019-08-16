# Colorization

Code to train and run a Colorization U-Net.

# Prerequisites

Machine needs a Nvidia GPU with installed drivers and CUDA 10.0 (for Tensorflow 1.14).  
Otherwise install the non gpu version of tensorflow and work on the CPU.

- Create a virtual environment: `virtualenv -p python3.7 <venv>` 
- Install the dependencies: `pip install -r requirements.txt`

# Training

Images and validation images need to be in the specified directory or in a direct subdirectory.  
The default models were trained with the option "large" and "alt_model".

```
usage: train.py [-h] --images IMAGES --val_images VAL_IMAGES --log_dir LOG_DIR
                [--weights WEIGHTS] [--image_size IMAGE_SIZE]
                [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--plot PLOT]
                [--loss LOSS] [--large] [--alt_model]

optional arguments:
  -h, --help            show this help message and exit
  --images IMAGES
  --val_images VAL_IMAGES
  --log_dir LOG_DIR
  --weights WEIGHTS
  --image_size IMAGE_SIZE
  --epochs EPOCHS
  --batch_size BATCH_SIZE
  --plot PLOT           Write image of model architecture to this png file
  --loss LOSS
  --large
  --alt_model
```

# Testing

The two notebooks Test_Directory and testing_notebook can be used to run inference.

Test_Directory espacially can be used to run inference on a whole directory of images. 
The Results are visualized next to the original and gray-scale image.

The testing_notebooks gives further insides in single images. You can visualize the predicted
a and b channels.

# Histograms

The Histogram notebook makes histograms from a directory of images. Number of images is currently 
limited by RAM of the running computer.