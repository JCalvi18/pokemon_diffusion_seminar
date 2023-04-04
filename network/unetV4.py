import torch
from torch import nn
from .positional import SinusoidalPositionEmbeddings
from .attention import LinearAttention, AttentionBlock
from .resNet import MinResnetBlock, ResAttention


class EncoderBlock(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, time_emb_dim, last_block=False):
        """
        Encoder block, composed of a minimal resnet block, a minimal attention head and a down sample module
        """
        super().__init__()
        self.res = MinResnetBlock(in_channel, out_channel, time_emb_dim)
        self.down = nn.Sequential(
            # Dimensions scale down by a factor of 2
            nn.MaxPool2d(2, stride=2, padding=0) if not last_block else nn.Identity(),
        )

    def forward(self, inputs, t):
        """
        :param inputs: [B,C,H,H]
        :param t: time embeddings
        :return (features after convolution, down sampled of features)
        """
        x = self.res(inputs, t)
        p = self.down(x)
        return x, p


class DecoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel, time_emb_dim, last_block=False):
        super().__init__()
        self.res = MinResnetBlock(out_channel *2, out_channel, time_emb_dim)
        self.attention = AttentionBlock(out_channel, out_channel, out_channel // 2)
        self.up = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2,
                                     padding=0) if not last_block else nn.Conv2d(in_channel, out_channel, 3, 1,
                                                                                 padding=1)

    def forward(self, inputs, skip_features, t):
        """
        :param inputs: down sampled input [B,C,W,D]
        :param skip_features: list features from encoder before down sample corresponding to the first resblock and
        after the attention
        :param t: Time embedding
        """
        x = self.up(inputs)
        x = self.attention(x, skip_features)
        x = torch.cat([x, skip_features], dim=1)
        x = self.res(x, t)
        return x


class Unet(nn.Module):
    """
    Bigger architecture with double resnet modules
    """

    def __init__(self, in_channel=4, out_channel=4):
        super().__init__()
        time_emb_dim = 32
        initial_channel_scale = 32

        # time embeddings
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
        )

        self.init_conv = nn.Conv2d(in_channel, initial_channel_scale, 1, padding=0)

        # Encoder
        # In -> [B,32,256,256] Out -> [B,64,128,128]
        self.e1 = EncoderBlock(initial_channel_scale, 64, time_emb_dim)

        # In -> [B,64,128,128] Out -> [B,128,64,64]
        self.e2 = EncoderBlock(64, 128, time_emb_dim)

        # In -> [B,128,64,64] Out -> [B,256,32,32]
        self.e3 = EncoderBlock(128, 256, time_emb_dim)

        # In -> [B,256,32,32] Out -> [B,512,32,32]
        self.e4 = EncoderBlock(256, 512, time_emb_dim, last_block=False)

        # Bridge
        # In -> [B,512,32,32] Out -> [B,512,32,32]
        self.b1 = MinResnetBlock(512, 1024, time_emb_dim)
        self.b2 = MinResnetBlock(1024, 1024, time_emb_dim)

        # Decoder
        # In -> [B,512,32,32] Out -> [B,256,32,32]
        self.d1 = DecoderBlock(1024, 512, time_emb_dim)

        # In -> [B,256,32,32] Out -> [B,128,64,64]
        self.d2 = DecoderBlock(512, 256, time_emb_dim)

        # In -> [B,128,64,64] Out -> [B,64,128,128]
        self.d3 = DecoderBlock(256, 128, time_emb_dim)

        # In -> [B,64,128,128] Out -> [B,32,256,256]
        self.d4 = DecoderBlock(128, 64, time_emb_dim, last_block=False)

        # Two times because we will concatenate with a skip connection os the same size
        self.last_res = MinResnetBlock(64 , initial_channel_scale, time_emb_dim)
        self.last_conv = nn.Conv2d(initial_channel_scale*2, out_channel, 1)

    def forward(self, inputs, timestep):
        # Encoder
        t = self.time_mlp(timestep)

        x = self.init_conv(inputs)
        in_skip = x.clone()

        s1, p1 = self.e1(x, t)
        s2, p2 = self.e2(p1, t)
        s3, p3 = self.e3(p2, t)
        s4, p4 = self.e4(p3, t)

        # Bridge
        b = self.b1(p4, t)
        b = self.b2(b, t)

        # Decoder
        d1 = self.d1(b, s4, t)
        d2 = self.d2(d1, s3, t)
        d3 = self.d3(d2, s2, t)
        d4 = self.d4(d3, s1, t)

        x = self.last_res(d4, t)
        x = torch.cat([x, in_skip], dim=1)

        return self.last_conv(x)
