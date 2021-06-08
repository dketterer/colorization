import argparse
import os
import sys

import torch

import colorization.infer as infer
import colorization.train as train
from colorization.chkpt_utils import get_latest
from colorization.model import Model


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--verbose', '-v', action='store_true')

    subparsers = parser.add_subparsers(help='sub-command', dest='command')
    subparsers.required = True

    parser_train = subparsers.add_parser('train', help='train a network')
    parser_train.add_argument('model', metavar='path', type=str,
                              help='path to output model or checkpoint to resume from')
    parser_train.add_argument('transform', metavar='path', type=str, help='path transform.py')
    parser_train.add_argument('--images', metavar='path', type=str, help='path to images', default='.')
    parser_train.add_argument('--val_images', metavar='path', type=str, help='path to images')
    parser_train.add_argument('--segment_masks', metavar='path', type=str, help='path to images')
    parser_train.add_argument('--val_segment_masks', metavar='path', type=str, help='path to images')
    parser_train.add_argument('--backbone', action='store', type=str, help='backbone model', default='UNet_bc64_d4')
    parser_train.add_argument('--head_type', type=str, help='head type', default='regression')
    parser_train.add_argument('--loss', metavar='selection', type=str,
                              choices=['L2', 'L1', 'L1+CCL', 'L2+CCL', 'L2W', 'L1W', 'L1W+CCL', 'L2W+CCL', 'L1+CCL-gt',
                                       'L1W+CCL-gt', 'L2+CCL-gt', 'L2W+CCL-gt'],
                              default='L2')
    parser_train.add_argument('--lambda_ccl', metavar='value', type=float, default=0.5)
    parser_train.add_argument('--ccl_version', metavar='selection', type=str, choices=['linear', 'euclidean', 'square'],
                              default='linear')
    parser_train.add_argument('--lr', metavar='value', type=float, help='Learning rate', default=0.0003)
    parser_train.add_argument('--alpha', metavar='value', type=float, help='Weighting factor alpha', default=5)
    parser_train.add_argument('--gamma', metavar='value', type=float, help='Weighting factor gamma', default=.5)
    parser_train.add_argument('--regularization_l2', '-reg_l2', metavar='value', type=float,
                              help='Weight regularization', default=0.)
    parser_train.add_argument('--momentum', metavar='value', type=float, help='SGD Optimizer Momentum', default=0.9)
    parser_train.add_argument('--optimizer', metavar='selection', help='The Optimizer', type=str, default='adam',
                              choices=['adam', 'sgd'])
    parser_train.add_argument('--iterations', '-iters', metavar='value', type=int, help='How many mini batches',
                              required=True)
    parser_train.add_argument('--val_iterations', '-val_iters', metavar='value', type=int,
                              help='Validation run after how many mini batches', default=3000)
    parser_train.add_argument('--warmup', metavar='iterations', help='numer of warmup iterations', type=int,
                              default=1000)
    parser_train.add_argument('--milestones', metavar='iterations', action='store', type=int, nargs='*',
                              help='List of iteration indices where learning rate decays', default=[])
    parser_train.add_argument('--growing_parameters', metavar='path', type=str,
                              help='Json file with the params fpr batch size and image size', required=True)
    parser_train.add_argument('--debug', action='store_true', help='No shuffle')

    parser_infer = subparsers.add_parser('infer', help='run inference on data')
    parser_infer.add_argument('model', type=str, help='path to output model or checkpoint to resume from')
    parser_infer.add_argument('--images', metavar='path', type=str, help='path to images', default='.')
    parser_infer.add_argument('--target_path', metavar='path', type=str, help='path to images', default='predictions')
    parser_infer.add_argument('--batch_size', metavar='value', type=int, help='Batch size', default=8)
    parser_infer.add_argument('--img_limit', metavar='value', type=int, help='Only first N images in the folder',
                              default=50)
    parser_infer.add_argument('--debug', action='store_true', help='Stich original image, grey and predicted together')

    parsed_args = parser.parse_args(args)
    return parsed_args


def load_model(args, verbose=False):
    model_path = get_latest(args.model)
    if args.command != 'train' and not model_path:
        raise RuntimeError('Model file {} does not exist!'.format(args.model))

    _, ext = os.path.splitext(args.model)

    if args.command == 'train' and not model_path:
        if verbose:
            print('Initializing model...')
        model = Model(backbone_name=args.backbone, head_type=args.head_type)
        state = {'path': args.model}
        model.initialize()
        if verbose: print(model)

    elif ext == '.pth' or ext == '.torch':
        if verbose: print('Loading model from {}...'.format(os.path.basename(model_path)))
        model, state = Model.load(model_path)
        state.update({'path': args.model})
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
        train.train(model=model, state=state, train_data_path=args.images, val_data_path=args.val_images,
                    transform_file=args.transform, growing_parameters=args.growing_parameters,
                    optimizer_name=args.optimizer, lambda_ccl=args.lambda_ccl, loss_type=args.loss,
                    ccl_version=args.ccl_version, train_segment_masks_path=args.segment_masks,
                    val_segment_masks_path=args.val_segment_masks,
                    lr=args.lr, alpha=args.alpha, gamma=args.gamma,
                    regularization_l2=args.regularization_l2, iterations=args.iterations,
                    val_iterations=args.val_iterations, milestones=args.milestones,
                    warmup=args.warmup, verbose=args.verbose, debug=args.debug)

    elif args.command == 'infer':
        infer.infer(model=model,
                    image_path=args.images,
                    target_path=args.target_path,
                    batch_size=args.batch_size,
                    img_limit=args.img_limit,
                    transform=None,
                    debug=args.debug)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
