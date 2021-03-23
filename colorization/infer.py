import torch

from colorization.model import Model


def infer(model: Model,
          image_path: str):
    model = model.eval()
    torch.no_grad()
