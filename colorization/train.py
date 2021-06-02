import copy
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from colorization.loss import L2Loss, L1Loss, L2CCLoss, L1CCLoss, ColorConsistencyLoss

np.random.seed(0)
import torch

torch.manual_seed(0)
import cv2

from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torchvision.transforms import ToTensor, Compose, Normalize

from colorization.data import ImagenetData, ImagenetColorSegmentData, get_trainloader, SavableShuffleSampler
from colorization.chkpt_utils import delete_older_then_n
from colorization.model import Model
from colorization.preprocessing import to_tensor_l, to_tensor_ab
from colorization.infer import infer
from colorization.metrics import PSNR, SSIM, PSNR_RGB


def imshow(img):
    img = (img / 2 + 0.5) * 255  # unnormalize
    img = torchvision.utils.make_grid(img)
    npimg = img.numpy()
    # npimg[0, ...] *= 255.
    # print(np.min(npimg[0, ...]), np.max(npimg[0, ...]))
    # print(np.min(npimg[1, ...]), np.max(npimg[1, ...]))
    # print(np.min(npimg[2, ...]), np.max(npimg[2, ...]))
    # npimg = npimg.reshape(3, npimg.shape[2], npimg.shape[0] * npimg.shape[3])
    npimg = np.round(npimg).astype(np.uint8)
    npimg = cv2.cvtColor(np.transpose(npimg, (1, 2, 0)), cv2.COLOR_LAB2RGB)
    plt.imshow(npimg)
    # plt.imshow(npimg[..., 0])
    plt.show()


# TODO gleiche methode wie training
def get_validation_metrics(validationloader, model, criterion, ccl_version='linear'):
    model = model.eval()
    metrics_results = torch.zeros(5).cuda()

    ssim = SSIM(window_size=11, gaussian_weights=True).cuda()
    ssim_uniform = SSIM(window_size=7, gaussian_weights=False).cuda()
    psnr = PSNR()
    # psnr_rgb = PSNR_RGB()
    for i, data in enumerate(tqdm(validationloader, leave=False, desc='Validation')):
        if i == 5000:
            break

        if len(data) == 3:
            inputs, labels, segment_masks = data
            inputs, labels, segment_masks = inputs.cuda(non_blocking=True), labels.cuda(
                non_blocking=True), segment_masks.cuda(non_blocking=True)
        else:
            inputs, labels = data
            inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            segment_masks = None

        with autocast():
            outputs = model(inputs)

            metrics_results[0] += criterion(outputs, labels, segment_masks)[0]
            metrics_results[1] += psnr(outputs, labels)
            # metrics_results[1] += psnr_rgb(outputs, labels, inputs)
            metrics_results[2] += ssim(outputs, labels)
            metrics_results[3] += ssim_uniform(outputs, labels)
            if segment_masks is not None:
                metrics_results[4] += ColorConsistencyLoss(ccl_version).cuda()(outputs, segment_masks)

        del outputs
        del labels
        del inputs
        if segment_masks is not None:
            del segment_masks

    metrics_results = metrics_results.cpu()
    metrics_results /= i

    loss_names = [criterion.__class__.__name__, 'PSNR', 'SSIM', 'SSIM-uniform', f'ColorConsistencyLoss-{ccl_version}'] \
        if metrics_results[4] != 0.0 else \
        [criterion.__class__.__name__, 'PSNR', 'SSIM', 'SSIM-uniform']
    avg_loss_values = metrics_results.numpy().tolist()
    return {name: avg_loss_values[i] for i, name in enumerate(loss_names)}


def fill_growing_parameters(sparse, iterations):
    assert 0 in sparse, 'Invalid growing parameters, missing iteration 0'
    filled = copy.deepcopy(sparse)
    prev = filled[0]
    for i in range(iterations):
        if i not in filled:
            filled[i] = prev
        else:
            prev = filled[i]
    return filled


def load_growing_parameters(path):
    with open(path, 'r') as f:
        loaded_params = json.load(f)
    spares_params = {int(k): (v[0], (v[1], v[2])) for k, v in loaded_params.items()}

    return spares_params


def train(model: Model,
          state: dict,
          train_data_path: str,
          val_data_path: str,
          transform_file: str,
          growing_parameters: dict,
          lr: float,
          iterations: int,
          val_iterations: int,
          verbose: bool,
          train_segment_masks_path: str = '',
          val_segment_masks_path: str = '',
          lambda_ccl=0.0,
          loss_type='L2',
          ccl_version='linear',
          alpha=5,
          gamma=.5,
          regularization_l2: float = 0.,
          warmup=5000,
          milestones=[],
          optimizer_name: str = 'sgd',
          print_every: int = 250,
          debug=False):
    model.train()
    torch.backends.cudnn.benchmark = True

    if debug:
        print_every = 10

    sparse_growing_parameters = load_growing_parameters(growing_parameters)
    filled_growing_parameters = fill_growing_parameters(sparse_growing_parameters, iterations)

    assert os.path.isfile(transform_file)
    sys.path.insert(0, os.path.dirname(transform_file))
    transforms = __import__(os.path.splitext(os.path.basename(transform_file))[0])

    trainset = ImagenetData(train_data_path, transform=None, transform_l=to_tensor_l, transform_ab=to_tensor_ab)
    testset = ImagenetData(val_data_path, transform=transforms.get_val_transform(1024), transform_l=to_tensor_l,
                           transform_ab=to_tensor_ab)
    if train_segment_masks_path or val_segment_masks_path:
        trainset = ImagenetColorSegmentData(train_data_path, train_segment_masks_path, transform=None,
                                            transform_l=to_tensor_l, transform_ab=to_tensor_ab)
        testset = ImagenetColorSegmentData(val_data_path, val_segment_masks_path,
                                           transform=transforms.get_val_transform(1024), transform_l=to_tensor_l,
                                           transform_ab=to_tensor_ab)

    model_dir = os.path.dirname(state['path'])

    writer = SummaryWriter(log_dir=os.path.join(model_dir, 'logs'))

    if loss_type == 'L2':
        criterion = L2Loss(weighted=False)
    elif loss_type == 'L2W':
        criterion = L2Loss(weighted=True, alpha=alpha, gamma=gamma)
    elif loss_type == 'L1':
        criterion = L1Loss(weighted=False)
    elif loss_type == 'L1W':
        criterion = L1Loss(weighted=True, alpha=alpha, gamma=gamma)
    elif loss_type == 'L2+CCL':
        criterion = L2CCLoss(lambda_ccl=lambda_ccl, ccl_version=ccl_version)
    elif loss_type == 'L2W+CCL':
        criterion = L2CCLoss(lambda_ccl=lambda_ccl, ccl_version=ccl_version, weighted=True, alpha=alpha, gamma=gamma)
    elif loss_type == 'L1+CCL':
        criterion = L1CCLoss(lambda_ccl=lambda_ccl, ccl_version=ccl_version)
    elif loss_type == 'L1W+CCL':
        criterion = L1CCLoss(lambda_ccl=lambda_ccl, ccl_version=ccl_version, weighted=True, alpha=alpha, gamma=gamma)
    else:
        raise NotImplementedError()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=regularization_l2)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=regularization_l2, momentum=0.9)
    else:
        raise NotImplementedError(f'Optimizer {optimizer_name} not available')
    if 'optimizer' in state:
        print('loading optimizer...')
        optimizer.load_state_dict(state['optimizer'])

    scaler = GradScaler(enabled=True)
    if 'scaler' in state:
        print('loading scaler...')
        scaler.load_state_dict(state['scaler'])

    def schedule(train_iter):
        if warmup and train_iter <= warmup:
            return 0.9 * train_iter / warmup + 0.1
        return 0.1 ** len([m for m in milestones if m <= train_iter])

    scheduler = LambdaLR(optimizer, schedule)
    if 'scheduler' in state:
        print('loading scheduler...')
        scheduler.load_state_dict(state['scheduler'])
    iteration = state.get('iteration', 0)

    sampler = SavableShuffleSampler(trainset, shuffle=not debug)
    if 'sampler' in state:
        print('loading sampler...')
        sampler.load_state_dict(state['sampler'])

    print(f'        Loss: {loss_type}')
    print(f'   Optimizer: {optimizer.__class__.__name__}')
    print(f'   Iteration: {iteration}/{iterations}')
    print(f'      Warmup: {warmup}')
    print(f'  Milestones: {milestones}')
    print(f'     Growing: {sparse_growing_parameters}')
    print(f' Sampler idx: {sampler.index}')
    print(f'Current step: {scheduler._step_count}')

    batch_size, input_size = filled_growing_parameters[iteration]
    trainset.transform = transforms.get_transform(input_size[0])
    trainloader = get_trainloader(trainset, batch_size, sampler)

    running_psnr, img_per_sec = 0.0, 0.0
    running_loss, avg_running_loss = defaultdict(float), defaultdict(float)
    tic = time.time()
    changed_batch_size = True
    psnr = PSNR()
    pbar = tqdm(total=iterations, initial=iteration)
    while iteration < iterations:
        loss_str = ' - '.join([f'{key}: {val:.5f} ' for key, val in avg_running_loss.items()])
        pbar.set_description(
            f'[Ep: {sampler.epoch} | B: {batch_size} | Im: {input_size[0]}x{input_size[1]}]  loss: {loss_str} - {img_per_sec:.2f} img/s')
        for data in trainloader:
            if iteration in sparse_growing_parameters and not changed_batch_size:
                # change batch size and input size
                batch_size, input_size = sparse_growing_parameters[iteration]
                trainset.transform = transforms.get_transform(input_size[0])
                # recreate the loader, otherwise the transform is not propagated in multiprocessing to the workers
                trainloader = get_trainloader(trainset, batch_size, sampler)
                changed_batch_size = True
                break
            else:
                changed_batch_size = False

            # get data
            if len(data) == 3:
                inputs, labels, segment_masks = data
                inputs, labels = inputs.cuda(non_blocking=True), (
                    labels.cuda(non_blocking=True), segment_masks.cuda(non_blocking=True))
            else:
                inputs, labels = data
                inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            with autocast():
                outputs = model(inputs)
                crit_labels = [*labels] if train_segment_masks_path else [labels]
                loss, loss_dict = criterion(outputs, *crit_labels)
                _psnr = psnr(outputs, labels[0] if train_segment_masks_path else labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

            del outputs
            del inputs
            del labels
            del data

            # print statistics
            for k, v, in loss_dict.items():
                running_loss[k] += v.item()
            running_psnr += _psnr.item()

            iteration += 1

            if iteration % print_every == 0 or iteration == iterations:
                img_per_sec = print_every * batch_size / (time.time() - tic)

                for k, v in running_loss.items():
                    avg_running_loss[k] = running_loss[k] / print_every
                    writer.add_scalar(f'train/{k}', avg_running_loss[k], global_step=iteration)
                avg_running_psnr = running_psnr / print_every

                writer.add_scalar('train/PSNR', avg_running_psnr, global_step=iteration)

                writer.add_scalar('Performance/Images per second', img_per_sec, global_step=iteration)
                writer.add_scalar('Learning rate', optimizer.param_groups[0]['lr'], global_step=iteration)
                if loss_type in ['L1+CCL', 'L2+CCL']:
                    writer.add_scalar('Parameters/lambda CCL', lambda_ccl, global_step=iteration)
                loss_str = ' - '.join([f'{key}: {val:.3f} ' for key, val in avg_running_loss.items()])
                pbar.set_description(
                    f'[Ep: {sampler.epoch} | B: {batch_size} | Im: {input_size[0]}x{input_size[1]}] loss: {loss_str} - {img_per_sec:.2f} img/s')

                running_loss = defaultdict(float)
                running_psnr = 0.0
                state.update({
                    'iteration': iteration,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                    'sampler': sampler.state_dict()
                })

                model.save(state, iteration)
                delete_older_then_n(state['path'], 10)

                tic = time.time()
            if iteration == iterations or iteration % val_iterations == 0:
                # run validation
                tic = time.time()
                torch.backends.cudnn.benchmark = False
                model = model.eval()
                test_loader = DataLoader(testset,
                                         batch_size=1,
                                         shuffle=False,
                                         num_workers=8,
                                         pin_memory=True,
                                         prefetch_factor=1)
                with torch.no_grad():
                    metric_results = get_validation_metrics(test_loader, model, criterion, ccl_version=ccl_version)
                for k, v in metric_results.items():
                    writer.add_scalar(f'validation/{k}', v, global_step=iteration)

                predicted_images = infer(model=model,
                                         image_path=val_data_path,
                                         target_path=os.path.join(model_dir, f'predictions-{iteration}'),
                                         batch_size=1,
                                         img_limit=20,
                                         transform=transforms.get_val_transform(1024),
                                         debug=True,
                                         tensorboard=True)
                for i, img in enumerate(predicted_images):
                    writer.add_image(f'example-{i}', img, global_step=iteration, dataformats='HWC')
                model = model.train()
                torch.backends.cudnn.benchmark = True
                tic = time.time()
            pbar.update(1)
            if iteration == iterations:
                break

    pbar.close()
    writer.close()
    print('Finished Training')
