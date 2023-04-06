import os
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode


class PokemonDataset (Dataset):
    """
    Custom map dataset that reads from folder path.
    """
    def __init__ (self, path='./pokemon', transform=None):
        self.path = path
        self.files_path = sorted (os.listdir (path))
        self.transform = transform

    def __len__ (self):
        return len (self.files_path)

    def __getitem__ (self, idx):
        img_path = os.path.join (self.path, self.files_path [idx])
        image = read_image (img_path, ImageReadMode.RGB)  # Read as RGBA chanels
        if self.transform:
            image = self.transform (image)
        return image