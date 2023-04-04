from torch import nn
import torch
import einops


class MultiHeadAttention(nn.Module):
    """
    Multi head attention block
    Based on https://arxiv.org/abs/1706.03762
    """
    def __init__(self, dim, heads=4, dim_head=32):
        """
        :param dim: Dimension of an image assuming it's square
        :param heads: number of heads
        """
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        """
        :param x: input (batch, color_channel, height, width)
        """
        b, c, h, w = x.shape
        #  Convolve input to a feature space representing queries, keys and values (batch, c_channel,
        qkv = self.to_qkv(x).chunk(3, dim=1)
        # Rearrange and get queries keys and values
        q, k, v = map(
            lambda t: einops.rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)
        out = einops.rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, in_channel, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(in_channel, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, in_channel, 1),
                                    nn.GroupNorm(1, in_channel))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: einops.rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = einops.rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class AttentionBlock (nn.Module):
    def __init__ (self, F_g, F_l, F_int):
        super (AttentionBlock, self).__init__ ()
        self.W_g = nn.Sequential (
            nn.Conv2d (F_g, F_int, kernel_size = 1, stride = 1, padding = 0, bias = True),
            nn.BatchNorm2d (F_int)
        )

        self.W_x = nn.Sequential (
            nn.Conv2d (F_l, F_int, kernel_size = 1, stride = 1, padding = 0, bias = True),
            nn.BatchNorm2d (F_int)
        )

        self.psi = nn.Sequential (
            nn.Conv2d (F_int, 1, kernel_size = 1, stride = 1, padding = 0, bias = True),
            nn.BatchNorm2d (1),
            nn.Sigmoid ()
        )

        self.relu = nn.ReLU (inplace = True)

    def forward (self, g, x):
        g1 = self.W_g (g)
        x1 = self.W_x (x)
        psi = self.relu (g1 + x1)
        psi = self.psi (psi)

        return x * psi