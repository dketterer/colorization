import argparse
import os

import keras
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler

from ImageDataGenerator import ImageDataGenerator
from callbacks import LRTensorBoard
from model.colorunet import l2_loss, l1_loss, smoothL1, ColorUNet
from keras.utils import plot_model
from keras_unet.models import custom_unet


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('images', type=str,
                        help='Read training images from this dir and its sub-dirs. Supported file exts: .jpg, .jpeg .JPG .JPEG')
    parser.add_argument('val_images', type=str,
                        help='Read validation images from this dir and its sub-dirs. Supported file exts: .jpg, .jpeg .JPG .JPEG')
    parser.add_argument('log_dir', type=str,
                        help='Log the tensorboard file, checkpoints, final weights and a trained model to this dir')
    parser.add_argument('loss', type=str, choices=['l1', 'l2'], help='The loss function')
    parser.add_argument('--weights', type=str,
                        help='Load this weight file and train on top of it. Read the epoch number and train the difference to "epochs"')
    parser.add_argument('--image_size', type=int, default=256,
                        help='One value for width and height of the resized training inputs')
    parser.add_argument('--epochs', type=int, default=4, help='Number of epochs the model shall be trained afterwards')
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--plot', type=str, help='Write image of the model architecture to this png file')
    parser.add_argument('--large', action='store_true', help='Use 64 filters in the first conv block not 32, and so on')
    parser.add_argument('--alt_model', action='store_true', help='Use the model from the keras-unet package instead of the self-defined')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.loss not in ['l1', 'l2']:
        print('arg loss must be in ', ['l1', 'l2'])
        exit(1)

    loss_fn = {'l1': l1_loss,
               'l2': l2_loss}[args.loss]

    batch_size = args.batch_size
    gpus = 1
    batch_size = batch_size * gpus
    image_size = (args.image_size, args.image_size)

    generator = ImageDataGenerator(folder=args.images, image_size=image_size, batch_size=batch_size, workers=12)
    val_generator = ImageDataGenerator(folder=args.val_images, image_size=image_size, batch_size=batch_size, workers=12)

    model = ColorUNet(input_size=(None, None, 1), large=args.large)
    if args.alt_model:
        filters = 32
        if args.large:
            filters = 64

        model = custom_unet(
            input_shape=(None, None, 1),
            use_batch_norm=True,
            upsample_mode='simple',
            num_classes=2,
            filters=filters,
            dropout=0.2,
            output_activation='tanh')

    if args.plot:
        plot_model(model, to_file=args.plot, show_shapes=True)
    model.summary()

    if gpus > 1:
        model = keras.utils.multi_gpu_model(model, gpus=gpus)
    model.compile(optimizer=Adam(lr=3e-4), loss=loss_fn, metrics=[smoothL1, 'mae', 'mse'])

    tensorboard = TensorBoard(log_dir=args.log_dir, batch_size=batch_size, write_images=True, write_grads=1)
    lrtensorboard = LRTensorBoard(log_dir=args.log_dir)
    ckptSaver = ModelCheckpoint(os.path.join(args.log_dir, 'weights.{epoch:02d}-{val_loss:.2f}.h5'), period=1)

    lr = [1e-3, 3e-4, 1e-4, 1e-4]
    schdeulelr = LearningRateScheduler(schedule=lambda idx: lr[idx])

    init_epochs = 0
    if args.weights:
        model.load_weights(args.weights)
        init_epochs = int(args.weights.split('.')[1].split('-')[0])

    history = model.fit_generator(generator.generate(batch_size),
                                  epochs=args.epochs,
                                  initial_epoch=init_epochs,
                                  validation_data=val_generator.generate(batch_size),
                                  steps_per_epoch=generator.size // batch_size,
                                  validation_steps=val_generator.size // batch_size,
                                  # optional add schdeulelr
                                  callbacks=[tensorboard, lrtensorboard, ckptSaver])

    model.save_weights(os.path.join(args.log_dir, 'trained_weights.h5'))
    model.save(os.path.join(args.log_dir, 'trained_model.hdf5'))

    generator.__del__()
    val_generator.__del__()
