import torch
from torch.optim import Adam
from model import Model
from network import SimpleUnet
from dataset import prepare_data
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

device = "cuda" if torch.cuda.is_available () else 'cpu'


def train_loop (args):
    # Read arguments
    total_timesteps = args.timesteps
    epochs = args.epochs
    lr = args.lr
    batch_size = args.batch
    dataset_path = args.dataset_path

    # Setup directory
    timedate_stamp = "{:%H-%M-%d}".format (datetime.now ())
    results_folder = Path (f"./results/{timedate_stamp}")
    results_folder.mkdir (parents = True, exist_ok = True)
    save_epoch_every = 10
    save_sample_every = 10

    network = SimpleUnet ().to (device)
    optimizer = Adam (network.parameters (), lr = lr)
    model = Model (network, total_timesteps)

    train_dataloader = prepare_data (dataset_path, batch_size)

    for epoch in tqdm (range (epochs)):
        for step, img_batch in tqdm (enumerate (train_dataloader), total = len (train_dataloader)):
            optimizer.zero_grad ()
            # Sample t uniformally for every sample in the batch
            t = torch.randint (0, total_timesteps, (batch_size,), device = device).long ()
            img_batch = img_batch.to (device)
            loss = model.train_step (img_batch, t, loss_type = 'huber')

            if step % save_sample_every == 0:
                with open (f'{results_folder}/loss.txt', 'a') as f:
                    print (f'Epoch: {epoch}, Sample: {step},  Loss: {loss.cpu ().detach ().numpy ():.5f}', file = f)
            loss.backward ()
            optimizer.step ()

            if epoch>0 and epochs % save_epoch_every == 0:
                model.save_model(results_folder, epoch)