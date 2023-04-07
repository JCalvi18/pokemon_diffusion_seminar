import torch
from model import Model
from network import UnetV1, UnetV2, UnetV3, UnetV4
from dataset import save_to_png, save_values
from pathlib import Path
from datetime import datetime

device = "cuda:0" if torch.cuda.is_available() else 'cpu'

unet_versions = [UnetV1, UnetV2, UnetV3, UnetV4]


def generate(args):
    load_path = args.load_path
    batch_size = args.batch
    total_timesteps = args.timesteps
    unet_version = args.unet_version
    offset = args.timestep_offset

    torch.manual_seed(args.seed)
    if device == 'cuda:0':
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True

    timedate_stamp = "{:%B-%d--%H:%M}".format(datetime.now())
    results_folder = Path(f"./results/gen/{timedate_stamp}")
    results_folder.mkdir(parents=True, exist_ok=True)
    network = unet_versions[unet_version]().to(device)
    model = Model(network, total_timesteps)

    model.load_model(load_path)

    output_dim = (batch_size, 3, 256, 256)
    results = model.inference_loop(output_dim)
    save_to_png(results_folder, results[-1])
    save_values(results_folder, results[-1])
    # Double denoise
    results = model.double_inference_loop(output_dim, offset)
    save_to_png(results_folder, results[-1], name='double_sample.png')
