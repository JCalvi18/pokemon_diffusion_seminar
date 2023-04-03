import torch
from torch.optim import Adam
from model import Model
from network import SimpleUnet, Unet
from dataset import prepare_data, save_to_png
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

device = "cuda:0" if torch.cuda.is_available() else 'cpu'


def train_loop(args):
    # Read arguments
    total_timesteps = args.timesteps
    epochs = args.epochs
    lr = args.lr
    batch_size = args.batch
    dataset_path = args.dataset_path

    # Set seed
    torch.manual_seed(args.seed)

    # Setup directory
    datetime_stamp = "{:%B-%d--%H:%M}".format(datetime.now())
    results_folder = Path(f"./results/train/{datetime_stamp}")
    results_folder.mkdir(parents=True, exist_ok=True)
    save_epoch_every = 10
    save_sample_every = 10
    with open(f'{results_folder}/summary.txt', 'a') as f:
        print(f'Total time steps: {total_timesteps}', file=f)
        print(f'Learning rate: {lr}', file=f)
        print(f'Epochs: {epochs}', file=f)
        print(f'Batch size: {batch_size}', file=f)

    network = Unet().to(device)
    optimizer = Adam(network.parameters(), lr=lr)
    model = Model(network, total_timesteps)

    train_dataloader = prepare_data(dataset_path, batch_size)

    for epoch in tqdm(range(epochs)):
        for step, img_batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=False):
            optimizer.zero_grad()
            # Sample t uniformally for every sample in the batch
            t = torch.randint(0, total_timesteps, (batch_size,), device=device).long()
            img_batch = img_batch.to(device)
            loss = model.train_step(img_batch, t, loss_type='huber')

            if step % save_sample_every == 0:
                with open(f'{results_folder}/loss.txt', 'a') as f:
                    print(f'Epoch: {epoch}, Sample: {step},  Loss: {loss.cpu().detach().numpy():.5f}', file=f)
            loss.backward()
            optimizer.step()

            if epoch > 0 and epochs % save_epoch_every == 0:
                model.save_model(results_folder, epoch)

    results_folder = Path(f"./results/gen/{datetime_stamp}/last")
    results_folder.mkdir(parents=True, exist_ok=True)

    output_dim = (4, 4, 256, 256)
    results = model.inference_loop(output_dim)
    save_to_png(results_folder, results[-1])
    results_folder = Path(f"./results/gen/{datetime_stamp}/init")
    results_folder.mkdir (parents = True, exist_ok = True)
    save_to_png(results_folder, results[0])

def generate(args):
    load_path = args.load_path
    batch_size = args.batch
    total_timesteps = args.timesteps

    torch.manual_seed(args.seed)
    timedate_stamp = "{:%B-%d--%H:%M}".format(datetime.now())
    results_folder = Path(f"./results/gen/{timedate_stamp}")
    results_folder.mkdir(parents=True, exist_ok=True)
    network = SimpleUnet().to(device)
    model = Model(network, total_timesteps)

    model.load_model(load_path)

    output_dim = (batch_size, 4, 256, 256)
    results = model.inference_loop(output_dim)
    save_to_png(results_folder, results[-1])