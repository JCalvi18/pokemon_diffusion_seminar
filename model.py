from utils import get_value_at_t, beta_scheduler, get_cumulative
from typing import Optional, Literal, Tuple
import torch
from torch import Tensor, randn
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path


class Model (object):
    """
    Class in charge of executing the forward and backward processes
    """

    def __init__ (self, network: torch.nn.Module, total_timesteps: int):
        self.total_timesteps = total_timesteps
        self.betas = beta_scheduler (total_timesteps)
        cumulative = get_cumulative (self.betas)
        # Calculate cumulative
        self.alphas_cumprod = cumulative [0]
        self.sqrt_alphas_cumprod = cumulative [1]
        self.sqrt_inv_alphas = cumulative [2]
        self.sqrt_one_minus_alphas_cumprod = cumulative [3]
        self.posterior_variance = cumulative [4]

        self.network: torch.nn.Module = network

    def forward_sample (self, x: Tensor, t: Tensor, noise: Tensor):
        """
        Forward pass, adding noise using the reparametrization trick
        :param x: input sample
        :param t: current timestep
        :param noise: noise to add in the input sample
        :return: noised image
        """

        if noise is None:
            noise = randn (x.shape, device = x.device)
        # Square root of cumulative alphas at t
        sqrt_cma_t = get_value_at_t (self.sqrt_alphas_cumprod, t, x.shape)
        # Square root of one minus cumulative alphas at t
        sqrt_one_minus_cma_t = get_value_at_t (self.sqrt_one_minus_alphas_cumprod, t, x.shape)

        # Section 3.2 reperametrized eq 4
        return sqrt_cma_t * x + sqrt_one_minus_cma_t * noise

    def train_step (self, x: Tensor, t: Tensor, loss_type: Literal ['l1', 'l2', 'huber'] = 'l1') -> Tensor:
        """
        Make a forward pass for the current batch, this entails adding different levels noise to the images
        and then calculating the difference w.r.t predicted noise of the network.
        :param x: batch of images
        :param t: batch of time steps from which to noise
        :param loss_type: metric to calculate loss
        :return: loss between noise used and predicted one by network
        """
        # Generate noise using Normal distribution
        noise = randn (x.shape, device = x.device)
        # Add noise for the different time steps
        noisy_x = self.forward_sample (x, t, noise)
        predicted_noise = self.network (noisy_x, t)

        if loss_type == 'l1':
            loss = F.l1_loss (noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss (noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss (noise, predicted_noise)
        else:
            raise NotImplementedError ()

        return loss

    @torch.no_grad ()
    def backward_sample (self, x: Tensor, t: Tensor, t_idx: int) -> Tensor:
        # Beta value at t
        betas_t = get_value_at_t (self.betas, t, x.shape)

        # Square root of one minus cumulative alphas at t
        sqrt_one_minus_cma_t = get_value_at_t (self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        # Square root of inverse alphas at t
        sqrt_ia_t = get_value_at_t (self.sqrt_inv_alphas, t, x.shape)

        predicted_noise = self.network (x, t)

        # Eq 11
        mean = sqrt_ia_t * (x - betas_t * predicted_noise / sqrt_one_minus_cma_t)

        if t_idx == 0:  # Last step
            return mean
        else:
            posterior_variance_t = get_value_at_t (self.posterior_variance, t, x.shape)
            # With 0 mean and unit variance
            noise = randn (x.shape, device = x.device)
            # Algorith 2 line 4
            return mean + torch.sqrt (posterior_variance_t) * noise

    def inference_loop (self, input_shape: Tuple) -> Tensor:
        self.network.eval ()
        device = next (self.network.parameters ()).device

        batches = input_shape [0]
        #  Init sample
        sample = randn (input_shape, device = device)

        result = []

        for i in tqdm (reversed (range (0, self.total_timesteps)), desc = 'Inference loop',
                       total = self.total_timesteps):
            # Consider array of same timesteps given that inference is done in batches
            timesteps = torch.full ((batches,), i, device = device, dtype = torch.long)
            sample = self.backward_sample (sample, timesteps, i)
            result.append (sample.cpu ())

        # Probably here there is a corruption of data
        return torch.cat(result, dim=0).reshape(((len(result),) + input_shape))
        # return result [-1]

    def save_model (self, results_folder, checkpoint: int):
        network_folder = Path (f"{results_folder}/network")
        network_folder.mkdir (parents = True, exist_ok = True)
        torch.save (self.network.state_dict (), f'{network_folder}/epoch-{checkpoint}.pth')

    def load_model (self, path: str):
        self.network.load_state_dict (torch.load (path))