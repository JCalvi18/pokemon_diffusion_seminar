import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class PokemonDataset (Dataset):
    """
    Custom map dataset (as defined by pytorch) that reads from folder path.
    """

    def __init__(self, path='./pokemon', augmentation=None,
                 resize=False, use_rgba=False):
        """
        Create a Pokemon Dataset
        :param path: Where the images reside
        :param augmentation: Composition of transformations to use as augmentation
        :param use_rgba: Whether to use transform the images as 3 or 4 channel tensors.
        """
        self.path = path
        self.files_path = sorted(os.listdir(path))
        self.augmentation = augmentation
        self.use_rgba = use_rgba
        # Normal transformations that allow us to use it in a model
        transformations = [
            transforms.ToTensor(),
            transforms.Lambda(lambda img: (img * 2) - 1),  # In range [-1,1]
        ]
        if resize:
            transformations.append(transforms.Resize((32, 32), antialias=True))

        self.input_transform = transforms.Compose(transformations)

    def __len__(self):
        return len(self.files_path)

    def __getitem__(self, idx, augmentation=True):
        """
        Map function, loads images lazily. Same implementation as the one provided by the supervisors.
        """
        img_path = os.path.join(self.path, self.files_path[idx])
        image = Image.open(img_path)
        out_image = Image.new("RGBA", image.size, "WHITE")
        out_image.paste(image, (0, 0), image)
        if not self.use_rgba:
            out_image = out_image.convert('RGB')
        if augmentation:
            return self.input_transform(self.augmentation(out_image))
        return self.input_transform(out_image)
