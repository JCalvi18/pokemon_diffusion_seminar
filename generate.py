import torch
from model import Model
from network import UnetV1, UnetV2, UnetV3, UnetV4
from dataset import save_to_png, save_values
from pathlib import Path
from datetime import datetime
from plot_utils import animate

device = "cuda:0" if torch.cuda.is_available() else 'cpu'

unet_versions = [UnetV1, UnetV2, UnetV3, UnetV4]


def generate(args):
    load_path = args.load_path
    batch_size = args.batch
    total_timesteps = args.timesteps
    unet_version = args.unet_version
    offset = args.timestep_offset
    resize_dataset = args.scale_down
    use_rgba = args.use_rgba
    image_size = 32 if resize_dataset else 256

    torch.manual_seed(args.seed)
    if device == 'cuda:0':
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True

    timedate_stamp = "{:%B-%d--%H:%M}".format(datetime.now())
    results_folder = Path(f"./results/gen/{timedate_stamp}")
    results_folder.mkdir(parents=True, exist_ok=True)

    channels = 4 if use_rgba else 3
    network = unet_versions[unet_version](channels, channels).to(device)
    model = Model(network, total_timesteps)

    model.load_model(load_path)

    output_dim = (batch_size, channels, image_size, image_size)
    results = model.inference_loop(output_dim)
    animate(results, 'backward', results_folder, fps=30)
    save_to_png(results_folder, results[-1])
    save_values(results_folder, results[-1])

    # Double denoise
    results = model.double_inference_loop(output_dim, offset)
    save_to_png(results_folder, results[-1], name='double_sample.png')