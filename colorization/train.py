import json
import os
import sys
import time
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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

from colorization.data import ImagenetData, get_trainloader
from colorization.chkpt_utils import delete_older_then_n
from colorization.model import Model
from colorization.preprocessing import to_tensor_l, to_tensor_ab
from colorization.infer import infer
from colorization.metrics import PSNR, SSIM, psnr_func


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


def get_validation_metrics(validationloader, model, metrics=[]):
    model = model.eval()
    metrics_results = torch.zeros(len(metrics)).cuda()
    for i, data in enumerate(tqdm(validationloader, leave=False, desc='Validation')):
        if i == 20000:
            break
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        with autocast():
            outputs = model(inputs)
            for j, metric in enumerate(metrics):
                metrics_results[j] += metric(outputs, labels)
            del outputs

    metrics_results = metrics_results.cpu()
    metrics_results /= i

    return metrics_results.numpy().tolist()


def fill_growing_parameters(sparse, iterations):
    assert 0 in sparse, 'Invalid growing parameters'
    prev = sparse[0]
    for i in range(iterations):
        if i not in sparse:
            sparse[i] = prev
        else:
            prev = sparse[i]
    return sparse


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
          regularization_l2: float = 0.,
          warmup=5000,
          milestones=[],
          optimizer_name: str = 'adam',
          print_every: int = 2,
          debug=False):
    model.train()

    sparse_growing_parameters = load_growing_parameters(growing_parameters)
    filled_growing_parameters = fill_growing_parameters(sparse_growing_parameters, iterations)

    assert os.path.isfile(transform_file)
    sys.path.insert(0, os.path.dirname(transform_file))
    transforms = __import__(os.path.splitext(os.path.basename(transform_file))[0])

    trainset = ImagenetData(train_data_path, transform=None, transform_l=to_tensor_l, transform_ab=to_tensor_ab)
    testset = ImagenetData(val_data_path, transform=transforms.get_val_transform(1024), transform_l=to_tensor_l,
                           transform_ab=to_tensor_ab)

    model_dir = os.path.dirname(state['path'])

    writer = SummaryWriter(log_dir=os.path.join(model_dir, 'logs'))

    if model.head_type == 'regression':
        criterion = torch.nn.MSELoss()
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
        optimizer.load_state_dict(state['optimizer'])

    scaler = GradScaler(enabled=True)
    if 'scaler' in state:
        scaler.load_state_dict(state['scaler'])

    def schedule(train_iter):
        if warmup and train_iter <= warmup:
            return 0.9 * train_iter / warmup + 0.1
        return 0.1 ** len([m for m in milestones if m <= train_iter])

    scheduler = LambdaLR(optimizer, schedule)
    if 'scheduler' in state:
        scheduler.load_state_dict(state['scheduler'])
    iteration = state.get('iteration', 0)
    epoch = state.get('epoch', 0)

    print(f'  Iteration: {iteration}/{iterations}')
    print(f'      Epoch: {epoch}')
    batch_size, input_size = filled_growing_parameters[iteration]
    trainset.transform = transforms.get_transform(input_size[0])
    trainloader = get_trainloader(trainset, batch_size, shuffle=not debug)

    running_loss, running_psnr, avg_running_loss, img_per_sec = 0.0, 0.0, 0.0, 0.0
    tic = time.time()
    changed = True
    pbar = tqdm(total=iterations, initial=iteration)
    while iteration < iterations:
        pbar.set_description(
            f'[Ep: {epoch} | B: {batch_size} | Im: {input_size[0]}x{input_size[1]}] loss: {avg_running_loss:.3f} - {img_per_sec:.2f} img/s')
        for data in trainloader:
            if iteration in sparse_growing_parameters and not changed:
                # change batch size and input size
                batch_size, input_size = sparse_growing_parameters[iteration]
                trainset.transform = transforms.get_transform(input_size[0])
                trainloader = get_trainloader(trainset, batch_size, shuffle=not debug)
                epoch += 1
                changed = True
                break
            else:
                changed = False

            # get data
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _psnr = psnr_func(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

            # print statistics
            running_loss += loss.item()
            running_psnr += _psnr.item()

            iteration += 1

            if iteration % print_every == 0 or iteration == iterations:
                img_per_sec = print_every * batch_size / (time.time() - tic)

                avg_running_loss = running_loss / print_every
                avg_running_psnr = running_psnr / print_every

                writer.add_scalar('train/MSE', avg_running_loss, global_step=iteration)
                writer.add_scalar('train/PSNR', avg_running_psnr, global_step=iteration)

                writer.add_scalar('Performance/Images per second', img_per_sec, global_step=iteration)
                writer.add_scalar('Learning rate', optimizer.param_groups[0]['lr'], global_step=iteration)
                pbar.set_description(
                    f'[Ep: {epoch} | B: {batch_size} | Im: {input_size[0]}x{input_size[1]}] loss: {avg_running_loss:.3f} - {img_per_sec:.2f} img/s')

                running_loss = 0.0
                running_psnr = 0.0
                state.update({
                    'epoch': epoch,
                    'iteration': iteration,
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict()
                })

                model.save(state, iteration)
                delete_older_then_n(state['path'], 10)

                tic = time.time()
            if iteration == iterations or iteration % val_iterations == 0:
                # run validation
                tic = time.time()
                model = model.eval()
                test_loader = DataLoader(testset,
                                         batch_size=1,
                                         shuffle=False,
                                         num_workers=4,
                                         pin_memory=True,
                                         prefetch_factor=1)
                metrics = [criterion, PSNR(), SSIM()]
                with torch.no_grad():
                    metric_results = get_validation_metrics(test_loader, model, metrics)
                for metric_result, metric in zip(metric_results, metrics):
                    writer.add_scalar(f'validation/{metric.__class__.__name__}', metric_result, global_step=iteration)

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
            pbar.update(1)

        epoch += 1
    pbar.close()
    writer.close()
    print('Finished Training')
