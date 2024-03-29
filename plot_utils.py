import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
import torch
import imageio


def plot_density(image):
    flattened_channel0 = image[:, :, 0].flatten()
    flattened_channel1 = image[:, :, 1].flatten()
    flattened_channel2 = image[:, :, 2].flatten()
    flattened_channel3 = image[:, :, 3].flatten()
    channels = [flattened_channel0, flattened_channel1,
                flattened_channel2, flattened_channel3]
    sns.kdeplot(data=channels)


def plot_forward(forward, results_folder=None):
    """
    :param forward: tensor [B,C,W,H]
    :param save: sa as file?
    :return:
    """
    np_transform = transforms.Compose([
        transforms.Lambda(lambda t: t.permute(0, 2, 3, 1)),  # BCHW to BHWC
        transforms.Lambda(lambda t: (t + 1) / 2),  # In range [0,1]
        transforms.Lambda(lambda t: t * 255.),  # In range [0,255]
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
    ])
    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(45, 15))
    fig.suptitle('Forward process', fontsize=40)
    transformed = [np_transform(img)[0] for img in forward]
    for i, row in enumerate(ax):
        for j, col in enumerate(row):
            col.set_title(f't={(5 * i + j) * 10}', fontsize=30)
            col.axis('off')
            col.imshow(transformed[i + j])

    # plt.tight_layout()
    if results_folder is not None:
        plt.savefig(f'{results_folder}/forward_grid.png', bbox_inches='tight')
    else:
        plt.show()


def plot_dataset(dataset, results_folder=None):
    np_transform = transforms.Compose([
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        transforms.Lambda(lambda t: (t + 1) / 2),  # In range [0,1]
        transforms.Lambda(lambda t: t * 255.),  # In range [0,255]
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
    ])
    images_per_row = 7
    fig, axes = plt.subplots(3, images_per_row, figsize=(45, 15))
    fig.suptitle('Dataset Samples and Augmentation', fontsize=32)
    # Plotting each image in a subplot
    for i, ax in enumerate(axes.flat):
        # The leftmost Pokemon is without augmentation.
        with_random_augmentation = False if i % images_per_row == 0 else True
        img = dataset.__getitem__(
            i//images_per_row, augmentation=with_random_augmentation)
        ax.imshow(np_transform(img))
        ax.axis('off')

    if results_folder is not None:
        plt.savefig(f'{results_folder}/dataset_summary.png',
                    bbox_inches='tight')
    else:
        plt.show()


def plot_color_distribution(args, model, dataset, train_dataloader,
                            device, forward=False, results_folder=None):
    # Create giant tensor with all images.
    img_num = len(dataset)
    img_size = 32 if args.scale_down else 256
    tensor_with_all_images = torch.zeros((img_num, 3, img_size, img_size))

    if forward:
        for i, imgs in enumerate(train_dataloader):
            t = torch.tensor(99, device=device).long()
            noisy = model.forward_sample(imgs, t.reshape(1))
            s = slice(i*args.batch, (i+1)*args.batch)
            tensor_with_all_images[s, :, :, :] = noisy
    else:
        for i, imgs in enumerate(train_dataloader):
            s = slice(i*args.batch, (i+1)*args.batch)
            tensor_with_all_images[s, :, :, :] = imgs

    # Save the pixel values of all images in 1D tensor for each channel.
    pixels_red = tensor_with_all_images[:, 0, :, :].flatten().numpy()
    pixels_green = tensor_with_all_images[:, 1, :, :].flatten().numpy()
    pixels_blue = tensor_with_all_images[:, 2, :, :].flatten().numpy()

    ax = sns.kdeplot(pixels_red, color="red", alpha=0.5, ls="--", lw=3)
    sns.kdeplot(pixels_green, color="green", ax=ax, alpha=0.5, lw=3)
    sns.kdeplot(pixels_blue, color="blue", ax=ax, alpha=0.5, ls=":", lw=3)
    if forward:
        plt.title("Color distribution of images after forward pass")
    else:
        plt.title("Color distribution of transformed original images")
    if results_folder is not None:
        name = 'color_distribution_normal_reference.png' if forward else 'color_distribution_original_images.png'        
        plt.savefig(f'{results_folder}/{name}',
                    dpi=300, bbox_inches='tight')
    else:
        plt.show()


def generate_forward(model, train_dataloader, timesteps):
    img = next(iter(train_dataloader))
    forward = [img]
    for t in timesteps:
        forward.append(model.forward_sample(img, t.reshape(1)))
    return forward


def animate(forward, name, results_folder=None, fps=10):
    np_transform = transforms.Compose([
        transforms.Lambda(lambda t: t.permute(0, 2, 3, 1)),  # BCHW to BHWC
        transforms.Lambda(lambda t: (t + 1) / 2),  # In range [0,1]
        transforms.Lambda(lambda t: t * 255.),  # In range [0,255]
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
    ])
    transformed = [np_transform(img)[0] for img in forward]
    if results_folder is not None:
        imageio.mimsave(f'{results_folder}/{name}.gif',
                        transformed, fps=fps)
    else:
        imageio.mimsave(f'{name}.gif',
                        transformed, fps=fps)

def show_image(array, results_folder=None):
    
    fig, ax = plt.subplots(nrows=2, ncols=2)
    fig.suptitle('Generated pokemons')
    idx = 0
    for i, row in enumerate(ax):
        for j, col in enumerate(row):
            # col.set_title(idx)
            col.axis('off')
            col.imshow(array[idx])
            idx += 1
    if results_folder is not None:
        plt.tight_layout()
        plt.savefig(f'{results_folder}/sample', bbox_inches='tight')
    plt.show()
    
    