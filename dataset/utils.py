from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from typing import List
import numpy as np
from .pokemonDataset import PokemonDataset

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use ('TkAgg')


def prepare_data (path, batch_size=64):
    """
    Construct input pipeline using custom dataset and transformations
    :param batch_size: size of each batch
    :return: dataloader object for training
    """
    input_transform = transforms.Compose ([
        transforms.RandomHorizontalFlip (),  # As stated on the paper
        transforms.Lambda (lambda img: img / 255),  # In range [0,1]
        transforms.Lambda (lambda img: (img * 2) - 1)  # In range [-1,1]
    ])

    dataset = PokemonDataset (path, transform = input_transform)
    train_dataloader = DataLoader (dataset, batch_size = batch_size, shuffle = True)
    return train_dataloader


np_transform = transforms.Compose ([
    transforms.Lambda (lambda t: (t + 1) / 2),  # In range [0,1]
    transforms.Lambda (lambda t: t.permute (1, 2, 0)),  # CHW to HWC
    transforms.Lambda (lambda t: t * 255.),  # In range [0,255]
    transforms.Lambda (lambda t: t.numpy ().astype (np.uint8)),
])


def output_to_image (t: Tensor):
    array = np_transform (t)
    return transforms.ToPILImage (array)


def show_image (img: Tensor):
    plt.imshow (np_transform (img))
    plt.show ()