import torch
from torch.optim import Adam
from model import Model
from network import MLP
from dataset import prepare_data

device = "cuda" if torch.cuda.is_available () else 'cpu'


def train_loop (args):
    total_timesteps = args.timesteps
    epochs = args.epochs
    lr = args.lr
    batch_size = args.batch

    save_and_sample_every =

    # TODO define network here
    network = MLP ()
    optimizer = Adam (network.parameters (), lr = lr)
    model = Model (network, total_timesteps)

    train_dataloader = prepare_data (batch_size)

    for epoch in range (epochs):
        for step, img_batch in enumerate (train_dataloader):
            optimizer.zero_grad ()
            # Sample t uniformally for every sample in the batch
            t = torch.randint (0, total_timesteps, (batch_size,), device = device).long ()
            img_batch = img_batch.to(device)
            loss = model.train_step(img_batch, t,loss_type = 'huber')

            if step % 100 == 0:
                print(loss)
            loss.backward()
            optimizer.step()