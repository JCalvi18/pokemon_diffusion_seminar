from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.io import write_video
import numpy as np
from .pokemonDataset import PokemonDataset
from pathlib import Path
import matplotlib.pyplot as plt


def prepare_data(path, batch_size=64, resize=False, return_dataset=False):
    """
    Construct input pipeline using custom dataset and transformations
    :param batch_size: size of each batch
    :return: dataloader object for training
    """

    augmentation = [
        transforms.RandomHorizontalFlip(),  # As stated on the paper
        # Lowest pixel becomes black and the lightest becomes white
        #transforms.RandomAutocontrast(),
        transforms.RandomRotation((-45, 45), fill=255),
        #transforms.ColorJitter(brightness=0.0, contrast=0.0, hue=0.2)
    ]

    augmentation = transforms.Compose(augmentation)

    dataset = PokemonDataset(path, augmentation=augmentation, resize=resize)
    train_dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return (train_dataloader, dataset) if return_dataset else train_dataloader


np_transform = transforms.Compose([
    transforms.Lambda(lambda t: t.permute(0, 2, 3, 1)),  # BCHW to BHWC
    transforms.Lambda(lambda t: (t + 1) / 2),  # In range [0,1]
    transforms.Lambda(lambda t: t * 255.),  # In range [0,255]
    transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
])

np_transform_video = transforms.Compose([
    transforms.Lambda(lambda t: (t + 1) / 2),  # In range [0,1]
    transforms.Lambda(lambda t: t.permute(1, 0, 3, 4, 2)),  # TBCHW to BTHWC
    transforms.Lambda(lambda t: t * 255.),  # In range [0,255]
    # transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
])


def output_to_image(t: Tensor):
    array = np_transform(t)
    return transforms.ToPILImage(array)


def output_to_video(path: str, t: Tensor):
    array = np_transform_video(t)
    for b, video in enumerate(array):
        write_video(f'{path}/pokemon-{b}.mp4', video, fps=30)


def save_to_png(path: Path, t: Tensor, name='sample.png'):
    array = np_transform(t)

    fig, ax = plt.subplots(nrows=2, ncols=2)
    fig.suptitle('Generated pokemons')
    # for b, img in enumerate(array):
    for i, row in enumerate(ax):
        for j, col in enumerate(row):
            col.set_title(i + j)
            col.axis('off')
            col.imshow(array[i + j])
    plt.tight_layout()
    plt.savefig(f'{path}/{name}', bbox_inches='tight')


def save_values(path: Path, t, name='values'):
    array = np_transform(t)
    np.save(f'{path}/{name}', array)


def show_image(img: Tensor):
    plt.imshow(np_transform(img))
    plt.show()