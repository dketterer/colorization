import argparse
import os
import sys

import torch

import colorization.infer as infer
import colorization.train as train
from colorization.model import Model


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--verbose', '-v', action='store_true')

    subparsers = parser.add_subparsers(help='sub-command', dest='command')
    subparsers.required = True

    parser_train = subparsers.add_parser('train', help='train a network')
    parser_train.add_argument('model', type=str, help='path to output model or checkpoint to resume from')
    parser_train.add_argument('transform', metavar='path', type=str, help='path transform.py')
    parser_train.add_argument('--images', metavar='path', type=str, help='path to images', default='.')
    parser_train.add_argument('--val_images', metavar='path', type=str, help='path to images', default='.')
    parser_train.add_argument('--backbone', action='store', type=str, help='backbone model', default='UNet_bc64_d4')
    parser_train.add_argument('--head_type', metavar='store', type=str, help='head type', default='regression')
    parser_train.add_argument('--lr', metavar='value', type=float, help='Learning rate', default=0.0003)
    parser_train.add_argument('--momentum', metavar='value', type=float, help='SGD Optimizer Momentum', default=0.9)
    parser_train.add_argument('--epochs', metavar='value', type=int, help='Epochs until end', default=1)

    parser_infer = subparsers.add_parser('infer', help='run inference on data')
    parser_infer.add_argument('model', type=str, help='path to output model or checkpoint to resume from')

    parsed_args = parser.parse_args(args)
    return parsed_args


def load_model(args, verbose=False):
    if args.command != 'train' and not os.path.isfile(args.model):
        raise RuntimeError('Model file {} does not exist!'.format(args.model))

    _, ext = os.path.splitext(args.model)

    if args.command == 'train' and not os.path.exists(args.model):
        if verbose:
            print('Initializing model...')
        model = Model(backbone_name=args.backbone, head_type=args.head_type)
        state = {'path': args.model}
        # TODO
        # model.initialize(args.fine_tune)
        if verbose: print(model)

    elif ext == '.pth' or ext == '.torch':
        if verbose: print('Loading model from {}...'.format(os.path.basename(args.model)))
        model, state = Model.load(args.model)
        if verbose: print(model)

    else:
        raise RuntimeError('Invalid model format "{}"!'.format(args.ext))

    return model, state


def main(args):
    model, state = load_model(args, verbose=args.verbose)
    # if model: model.share_memory()

    if torch.cuda.is_available():
        print(f'Use CUDA backend: {torch.cuda.get_device_name()}')

    if args.command == 'train':
        growing_parameters = {0: (512, (32, 32)),
                              8: (512, (64, 64)),
                              12: (256, (128, 128)),
                              16: (16, (256, 256)),
                              20: (8, (512, 512))}

        train.train(model, state, args.images, args.val_images, args.transform, growing_parameters, args.lr,
                    args.momentum, args.epochs, args.verbose)

    elif args.command == 'infer':
        infer.infer(model, args.images)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
