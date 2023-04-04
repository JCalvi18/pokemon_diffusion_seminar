import math
import torch
from torch import Tensor
from torch import nn


class SinusoidalPositionEmbeddings(nn.Module):
    """"
    Based on https://arxiv.org/abs/1706.03762

    """

    def __init__(self, dim):
        """
        :param dim: Dimension of an image assuming it's square
        """
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor):
        """
        :param t: Time step of noise level (batch_size, 1)
        :return: positional embedding (batch_size, dim)
        """
        device = t.device
        half_dim = self.dim // 2
        embeddings = torch.tensor([math.log(10000) / (half_dim - 1)], device=device)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
