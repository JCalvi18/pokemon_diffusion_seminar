import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class PokemonDataset (Dataset):
    """
    Custom map dataset that reads from folder path.
    """

    def __init__(self, path='./pokemon', augmentation=None, resize=False):
        self.path = path
        self.files_path = sorted(os.listdir(path))
        self.augmentation = augmentation
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
        img_path = os.path.join(self.path, self.files_path[idx])
        image = Image.open(img_path)
        rgba_image = Image.new("RGBA", image.size, "WHITE")
        rgba_image.paste(image, (0, 0), image)
        rgb_image = rgba_image.convert('RGB')
        if augmentation:
            return self.input_transform(self.augmentation(rgb_image))
        return self.input_transform(rgb_image)
