# Utils functions
import torch
import torch.nn.functional as F
from typing import Tuple


def beta_scheduler(timesteps: int, start=1e-4, end=2e-2):
    """
    Linear scheduler for beta
    :param timesteps: total number of timesteps
    :param start: intial beta value
    :param end: final beta value
    :return: tensor array
    """
    return torch.linspace(start, end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    x = torch.linspace(0, timesteps, timesteps + 1)
    alphas_cumprod = torch.cos(
        ((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def get_cumulative(betas: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate cumulative variables, for the sampling process
    :param betas: Tensor array, obtained using a scheduler
    :return: alphas_cumprod, sqrt_alphas_cumprod, sqrt_inv_alphas, sqrt_one_minus_alphas_cumprod, posterior_variance
    """
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    sqrt_inv_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # Add 1.0 to the beginning of the above array and remove last index
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

    # Square variance Section 3.2, 3rd line
    posterior_variance: torch.Tensor = betas * \
        (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    return (
        alphas_cumprod,
        sqrt_alphas_cumprod,
        sqrt_inv_alphas,
        sqrt_one_minus_alphas_cumprod,
        posterior_variance
    )


def get_value_at_t(var: torch.Tensor, timestep: torch.Tensor, sample_shape: tuple[int]):
    """
    Get the current parameters given a timestep array
    :param var: array from which to extract desired values
    :param timestep: array of time steps from which to extract
    :param sample_shape: shape of the sample to reshape output
    :return: parameters corresponding to the specified timestep from var
    """
    batch_size = timestep.shape[0]
    out = var.gather(-1, timestep.cpu())
    return out.reshape(batch_size, *((1,) * (len(sample_shape) - 1))).to(timestep.device)
