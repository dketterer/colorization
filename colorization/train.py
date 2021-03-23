import os
import sys
import time
import numpy as np

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
from colorization.model import Model
from colorization.preprocessing import to_tensor_l, to_tensor_ab
from colorization.infer import infer


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


def get_validation_loss(validationloader, model, criterion):
    running_loss = 0
    for i, data in enumerate(tqdm(validationloader)):
        inputs, labels = data
        labels = labels.permute(0, 3, 1, 2)
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss

    return running_loss / len(validationloader)


def fill_growing_parameters(sparse, epochs):
    assert 0 in sparse, 'Invalid growing parameters'
    prev = sparse[0]
    for i in range(epochs):
        if i not in sparse:
            sparse[i] = prev
        else:
            prev = sparse[i]
    return sparse


def train(model: Model,
          state: dict,
          train_data_path: str,
          val_data_path: str,
          transform_file: str,
          growing_parameters: dict,
          lr: float,
          momentum: float,
          epochs: int,
          verbose: bool,
          print_every: int = 30):
    batch_size = 4
    input_size = (512, 512)

    growing_parameters = fill_growing_parameters(growing_parameters, epochs)

    assert os.path.isfile(transform_file)
    sys.path.insert(0, os.path.dirname(transform_file))
    transforms = __import__(os.path.splitext(os.path.basename(transform_file))[0])

    transform = transforms.get_transform(input_size[0])

    trainset = ImagenetData(train_data_path, transform=transform, transform_l=to_tensor_l,
                            transform_ab=to_tensor_ab)
    trainloader = get_trainloader(trainset, batch_size)
    testset = ImagenetData(val_data_path, transform=transforms.get_val_transform(512), transform_l=to_tensor_l,
                           transform_ab=to_tensor_ab)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    if False:
        # get some random training images
        dataiter = iter(trainloader)
        images, labels = dataiter.next()
        # labels = labels.permute(0, 3, 1, 2)

        # show images
        imshow(torch.cat([images, labels], 1))
    if model.head_type == 'regression':
        criterion = torch.nn.MSELoss()
    else:
        raise NotImplementedError()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
    # lr: 0.0003
    # momentum = 0.9
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if 'optimizer' in state:
        optimizer.load_state_dict(state['optimizer'])

    scaler = GradScaler(enabled=True)
    start_epoch = state.get('epoch', -1) + 1
    for epoch in range(start_epoch, epochs):  # loop over the dataset multiple times
        # change batch size and input size
        if epoch in growing_parameters:
            batch_size, input_size = growing_parameters[epoch]
            trainset.transform = transforms.get_transform(input_size[0])
            trainloader = get_trainloader(trainset, batch_size)

        model = model.train()

        print(f'Start epoch {epoch + 1} with batch size: {batch_size} and image size: {input_size[0]}x{input_size[1]}')
        running_loss = 0.0
        tic = time.time()

        pbar = tqdm(trainloader)
        for i, data in enumerate(pbar):
            # get data
            inputs, labels = data
            # labels = labels.permute(0, 3, 1, 2)
            inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # print statistics
            running_loss += loss.item()

            if i % print_every == print_every - 1:  # print every 2000 mini-batches
                avg_running_loss = running_loss / print_every
                img_per_sec = print_every * batch_size / (time.time() - tic)
                pbar.set_description(
                    f'[{epoch + 1}, {i + 1}] loss: {running_loss:.3f} - {img_per_sec:.2f} img/s')
                tic = time.time()
                running_loss = 0.0
        # run validation
        # tic = time.time()
        # model = model.eval()
        # with torch.no_grad():
        #    val_loss = get_validation_loss(testloader, model, criterion)
        # model = model.train()
        # print(f'Validation in {(time.time() - tic):.2f}s - loss: {val_loss:.3f}')

        infer(model=model,
              image_path=val_data_path,
              target_path=os.path.join(os.path.dirname(state['path']), f'predictions-{epoch}'),
              batch_size=1,
              img_limit=20,
              debug=True)

        state.update({
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'loss': loss,
        })
        model.save(state)

    print('Finished Training')
