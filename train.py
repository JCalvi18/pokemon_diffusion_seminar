import torch
from torch.optim import Adam
from model import Model
from network import UnetV1, UnetV2, UnetV3, UnetV4
from dataset import prepare_data, save_to_png
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

# Detect the device
device = "cuda:0" if torch.cuda.is_available() else 'cpu'
# List unet version
unet_versions = [UnetV1, UnetV2, UnetV3, UnetV4]


def train_loop(args):
    # Read arguments
    total_timesteps = args.timesteps
    epochs = args.epochs
    lr = args.lr
    batch_size = args.batch
    dataset_path = args.dataset_path
    unet_version = args.unet_version
    resize_dataset = args.scale_down
    use_rgba = args.use_rgba

    # Set seed
    torch.manual_seed(args.seed)
    if device == 'cuda:0':
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
    # Setup directory
    datetime_stamp = "{:%B-%d--%H:%M}".format(datetime.now())
    results_folder = Path(f"./results/train/{datetime_stamp}")
    results_folder.mkdir(parents=True, exist_ok=True)
    save_epoch_every = 500
    save_sample_every = 10
    # Write file with arguments used
    with open(f'{results_folder}/summary.txt', 'a') as f:
        print(f'Total time steps: {total_timesteps}', file=f)
        print(f'Learning rate: {lr}', file=f)
        print(f'Epochs: {epochs}', file=f)
        print(f'Batch size: {batch_size}', file=f)
        print(f'U-Net version: {unet_version}', file=f)
        print(f'Scaled: {resize_dataset}', file=f)
        print(f'Using RGBA: {use_rgba}', file=f)

    # Define number of channels to use
    channels = 4 if use_rgba else 3

    # Initialize core modules
    network = unet_versions[unet_version](channels, channels).to(device)
    optimizer = Adam(network.parameters(), lr=lr)
    model = Model(network, total_timesteps)
    train_dataloader = prepare_data(dataset_path, batch_size,
                                    resize=resize_dataset, use_rgba=use_rgba)
    # Training Loop
    for epoch in tqdm(range(epochs)):
        for step, img_batch in tqdm(enumerate(train_dataloader),
                                    total=len(train_dataloader), leave=False):
            optimizer.zero_grad()
            # Sample time step uniformly for every sample in the batch
            t = torch.randint(0, total_timesteps,
                              (batch_size,), device=device).long()
            img_batch = img_batch.to(device)
            loss = model.train_step(img_batch, t, loss_type='huber')

            # Optimize
            loss.backward()
            optimizer.step()

            # Write to file or save if necessary
            if step % save_sample_every == 0:
                with open(f'{results_folder}/loss.txt', 'a') as f:
                    print(
                        f'Epoch: {epoch}, Sample: {step}, Loss: {loss.cpu().detach().numpy(): .5f}', file=f)
            if (epoch > 0 and epoch % save_epoch_every == 0) or epoch == epochs-1:
                model.save_model(results_folder, epoch+1)
