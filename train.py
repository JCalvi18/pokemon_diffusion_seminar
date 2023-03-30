import torch
from torch.optim import Adam
from model import Model
from network import SimpleUnet
from dataset import prepare_data
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available () else 'cpu'


def train_loop (args):
    total_timesteps = args.timesteps
    epochs = args.epochs
    lr = args.lr
    batch_size = args.batch
    dataset_path = args.dataset_path


    # TODO define network here
    network = SimpleUnet().to(device)
    optimizer = Adam (network.parameters (), lr = lr)
    model = Model (network, total_timesteps)

    train_dataloader = prepare_data (dataset_path, batch_size)

    for epoch in tqdm(range(epochs)):
        for step, img_batch in tqdm(enumerate(train_dataloader), total = len(train_dataloader)):
            optimizer.zero_grad ()
            # Sample t uniformally for every sample in the batch
            t = torch.randint (0, total_timesteps, (batch_size,), device = device).long ()
            img_batch = img_batch.to(device)
            loss = model.train_step(img_batch, t,loss_type = 'huber')

            if step % 10 == 0:
                print(f'Loss: {loss.cpu().detach().numpy()}')
            loss.backward()
            optimizer.step()