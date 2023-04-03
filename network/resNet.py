import torch
from torch import Tensor
from torch import nn


class ResnetBlock(nn.Module):

    def __init__(self, in_channels, out_channels, time_emb_dim, groups=8):
        """
        ReSnet block, implements two convolutions, group normalization and SiLU activation
        :param in_channels: Input channels
        :param out_channels: Output channels
        :param groups: Number of groups to use in group normalization
        """
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            # Out channels are the double because we are going to use the scale and shift technique
            nn.Linear(time_emb_dim, out_channels * 2)
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.SiLU()
        # Skip connection, if in and out channels differ use a convolution else just skip
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, inputs, t):
        x = self.conv1(inputs)
        x = self.norm(x)
        time_emb: Tensor = self.time_mlp(t)
        # Transform from BC to BCWH
        time_emb = time_emb[(...,) + (None,) * 2]
        # Split tensor into 2 given we have double the dimension
        scale, shift = time_emb.chunk(2, dim=1)
        x = x * (scale + 1) + shift
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm(x)
        x = self.activation(x)
        # return sum of previous operations and the skip connection
        return x + self.res_conv(inputs)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class ResAttention(nn.Module):
    """
    Apply skip connection to attention block, for attention block apply group normalization before forward pass
    """
    def __init__(self, in_channel, att_module: nn.Module):
        super().__init__()
        self.att_module = att_module
        self.norm = nn.GroupNorm(1, in_channel)

    def forward(self, in_sample):
        x = self.norm(in_sample)
        return self.att_module(x) + in_sample
