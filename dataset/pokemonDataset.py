import os
from torch.utils.data import Dataset
from PIL import Image

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
        image = Image.open (img_path)
        rgba_image = Image.new ("RGBA", image.size, "WHITE")
        rgba_image.paste (image, (0, 0), image)
        rgb_image = rgba_image.convert ('RGB')
        if self.transform:
            image = self.transform (rgb_image)
        return image