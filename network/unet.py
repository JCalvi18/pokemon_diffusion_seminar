import torch
from torch import nn
from .positional import SinusoidalPositionEmbeddings
from .attention import LinearAttention
from .resNet import ResnetBlock, ResAttention


class EncoderBlock(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, time_emb_dim, last_block=False):
        """
        Encoder block, composed of a resnet block, a linear attention head and a down sample module
        """
        super().__init__()
        self.res1 = ResnetBlock(in_channel, in_channel, time_emb_dim)
        self.res2 = ResnetBlock(in_channel, in_channel, time_emb_dim)
        self.attention = ResAttention(in_channel, LinearAttention(in_channel))
        self.down = nn.Sequential(
            # Dimensions don't change only the number of channels increases
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            # Here dimensions scale by a factor of 2
            nn.MaxPool2d(2, stride=2, padding=0) if not last_block else nn.Identity(),
        )

    def forward(self, inputs, t):
        """
        :param inputs: [B,C,H,H]
        :param t: time embeddings
        :return (features up until attention, down sampled of features)
        """
        skip_features = []
        # Pass it to two resnet blocks
        x = self.res1(inputs, t)
        skip_features.append(x)
        x = self.res2(x, t)
        x = self.attention(x)
        skip_features.append(x)
        p = self.down(x)
        return skip_features, p


class DecoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel, time_emb_dim, last_block=False):
        super().__init__()
        # in_channel size = size of down sampled features, out_channel size = size features before down sample
        # Given we are going to concatenate with the skip connections of the encoder block,
        # features without down sampling. Then, Input shape of the resnet block is = in_channel + out_channel
        self.res1 = ResnetBlock(in_channel + out_channel, in_channel, time_emb_dim)
        self.res2 = ResnetBlock(in_channel + out_channel, in_channel, time_emb_dim)
        self.attention = ResAttention(in_channel, LinearAttention(in_channel))
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
        x = torch.cat([inputs, skip_features[0]], dim=1)
        x = self.res1(x, t)
        x = torch.cat([x, skip_features[1]], dim=1)
        x = self.res2(x, t)
        x = self.attention(x)

        x = self.up(x)
        return x


class Unet(nn.Module):
    def __init__(self, in_channel=4, out_channel=4):
        super().__init__()
        time_emb_dim = 32
        initial_channel_scale = 16

        # time embeddings
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
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
        self.e4 = EncoderBlock(256, 512, time_emb_dim, last_block=True)

        # Bridge
        # In -> [B,512,32,32] Out -> [B,512,32,32]
        self.b1 = ResnetBlock(512, 512, time_emb_dim)
        self.b_att = ResAttention(512, LinearAttention(512))
        self.b2 = ResnetBlock(512, 512, time_emb_dim)

        # Decoder
        # In -> [B,512,32,32] Out -> [B,256,32,32]
        self.d1 = DecoderBlock(512, 256, time_emb_dim)

        # In -> [B,256,32,32] Out -> [B,128,64,64]
        self.d2 = DecoderBlock(256, 128, time_emb_dim)

        # In -> [B,128,64,64] Out -> [B,64,128,128]
        self.d3 = DecoderBlock(128, 64, time_emb_dim)

        # In -> [B,64,128,128] Out -> [B,32,256,256]
        self.d4 = DecoderBlock(64, initial_channel_scale, time_emb_dim, last_block=True)

        # Two times because we will concatenate with a skip connection os the same size
        self.last_res = ResnetBlock(initial_channel_scale * 2, initial_channel_scale, time_emb_dim)
        self.last_conv = nn.Conv2d(initial_channel_scale, out_channel, 1)

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
        b = self.b_att(b)
        b = self.b2(b, t)

        # Decoder
        d1 = self.d1(b, s4, t)
        d2 = self.d2(d1, s3, t)
        d3 = self.d3(d2, s2, t)
        d4 = self.d4(d3, s1, t)

        x = torch.cat([d4, in_skip], dim=1)
        x = self.last_res(x,t)

        return self.last_conv(x)
