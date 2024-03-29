from model_utils import get_value_at_t, beta_scheduler, get_cumulative
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

    def __init__(self, network: torch.nn.Module, total_timesteps: int):
        self.total_timesteps = total_timesteps
        # Define variables used in the paper
        self.betas = beta_scheduler(total_timesteps)
        cumulative = get_cumulative(self.betas)
        # Calculate cumulative
        self.alphas_cumprod = cumulative[0]
        self.sqrt_alphas_cumprod = cumulative[1]
        self.sqrt_inv_alphas = cumulative[2]
        self.sqrt_one_minus_alphas_cumprod = cumulative[3]
        self.posterior_variance = cumulative[4]

        self.network: torch.nn.Module = network

    def forward_sample(self, x: Tensor, t: Tensor, noise: Tensor = None):
        """
        Forward pass, adding noise using the reparametrization trick
        :param x: input sample
        :param t: current timestep
        :param noise: noise to add in the input sample
        :return: noised image
        """

        if noise is None:
            noise = randn(x.shape, device=x.device)
        # Square root of cumulative alphas at t
        sqrt_cma_t = get_value_at_t(self.sqrt_alphas_cumprod, t, x.shape)
        # Square root of one minus cumulative alphas at t
        sqrt_one_minus_cma_t = get_value_at_t(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape)

        # Section 3.2 reperametrized eq 4
        return sqrt_cma_t * x + sqrt_one_minus_cma_t * noise

    def train_step(self, x: Tensor, t: Tensor, loss_type: Literal['l1', 'l2', 'huber'] = 'l1') -> Tensor:
        """
        Make a forward pass for the current batch, this entails adding different levels noise to the images
        and then calculating the difference w.r.t predicted noise of the network.
        :param x: batch of images
        :param t: batch of time steps from which to noise
        :param loss_type: metric to calculate loss
        :return: loss between noise used and predicted one by network
        """
        # Generate noise using Normal distribution
        noise = randn(x.shape, device=x.device)
        # Add noise for the different time steps
        noisy_x = self.forward_sample(x, t, noise)
        predicted_noise = self.network(noisy_x, t)

        # Calculate loss depending on the type of loss
        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss

    @torch.no_grad()
    def backward_sample(self, x: Tensor, t: Tensor, t_idx: int) -> Tensor:
        """"
        Make a backward pass for a batch of samples, by calculating the applied noise at t-1
        :param x: batch of samples
        :param t: Current time steps corresponding to the batch of samples
        :param t_idx: Time step index to control whether we are at the last time step or not
        :return denoised version of the samples
        """
        # Beta value at t
        betas_t = get_value_at_t(self.betas, t, x.shape)

        # Square root of one minus cumulative alphas at t
        sqrt_one_minus_cma_t = get_value_at_t(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        # Square root of inverse alphas at t
        sqrt_ia_t = get_value_at_t(self.sqrt_inv_alphas, t, x.shape)

        predicted_noise = self.network(x, t)

        # Eq 11
        mean = sqrt_ia_t * \
            (x - betas_t * predicted_noise / sqrt_one_minus_cma_t)

        if t_idx == 0:  # Last step
            return mean
        else:
            posterior_variance_t = get_value_at_t(
                self.posterior_variance, t, x.shape)
            # With 0 mean and unit variance
            noise = randn(x.shape, device=x.device)
            # Algorith 2 line 4
            return mean + torch.sqrt(posterior_variance_t) * noise

    def inference_loop(self, input_shape: Tuple) -> Tensor:
        """
        Generate an image using by iterating over time steps and on each one denoise using the backward pass.
        :param input_shape: Shape of the type [B,C,W,H]
        :return Tensor of generated images at each times step of shape [T,B,C,W,H]
        """
        # Set the network on evaluation mode
        self.network.eval()
        # Setup
        device = next(self.network.parameters()).device
        batches = input_shape[0]

        #  Init sample, pure noise
        sample = randn(input_shape, device=device)
        result = []

        # Generating loop
        for i in tqdm(reversed(range(0, self.total_timesteps)), desc='Inference loop',
                      total=self.total_timesteps):
            # Consider array of same timesteps given that inference is done in batches
            timesteps = torch.full(
                (batches,), i, device=device, dtype=torch.long)
            # Get denoised samples
            sample = self.backward_sample(sample, timesteps, i)
            # Save samples at t
            result.append(sample.cpu())

        # Convert list to tensor
        return torch.cat(result, dim=0).reshape(((len(result),) + input_shape))

    def double_inference_loop(self, input_shape: Tuple, time_step_offset: int) -> Tensor:
        """
        Generate an image as the function above, but start the denoising
        procedure again using an already denoised sample
        :param input_shape: Shape of the type [B,C,W,H]
        :param time_step_offset: From which time step start to denoise again
        :return Tensor of generated images at each times step of shape [T,B,C,W,H]
        """
        # Set the network on evaluation mode
        self.network.eval()
        # Setup
        device = next(self.network.parameters()).device
        batches = input_shape[0]
        #  Init sample, pure random noise
        sample = randn(input_shape, device=device)

        result = []

        # Construct a tensor of time steps considering the ones to be repeated
        timesteps = torch.cat(
            [torch.arange(0, time_step_offset), torch.arange(0, self.total_timesteps)])
        # Generating loop
        for i in tqdm(reversed(timesteps), desc='Inference loop',
                      total=len(timesteps)):
            # Consider array of same timesteps given that inference is done in batches
            timesteps = torch.full(
                (batches,), i, device=device, dtype=torch.long)
            # Get denoised samples
            sample = self.backward_sample(sample, timesteps, i)
            result.append(sample.cpu())
        return torch.cat(result, dim=0).reshape(((len(result),) + input_shape))

    def save_model(self, results_folder, checkpoint: int):
        """
        :param results_folder: path where to save the model
        :param checkpoint: epoch to be saved
        """
        network_folder = Path(f"{results_folder}/network")
        network_folder.mkdir(parents=True, exist_ok=True)
        torch.save(self.network.state_dict(),
                   f'{network_folder}/epoch-{checkpoint}.pth')

    def load_model(self, path: str):
        self.network.load_state_dict(torch.load(path))
