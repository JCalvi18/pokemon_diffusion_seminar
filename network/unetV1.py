from torch import nn
import math
import torch

'''
The postion embeddings are used to allow the model to differ to differ between different timesteps. 
The implementation is from https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/
'''
class SinusoidalPositionEmbeddings (nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(
            half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

'''
The encoder block applies a adds a pooling layer to the operations of the conv-block.
'''
class EncoderBlock (nn.Module):
    def __init__(self, in_c, out_c, time_emb_dim):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c, time_emb_dim)
        self.pool = nn.MaxPool2d((2, 2))
        # see how to implement time embeddings

    def forward(self, inputs, t):
        x = self.conv(inputs, t)
        p = self.pool(x)
        return x, p


'''The Decoder Block alternates between a layers for upsampling and a convolutional layer.'''
class DecoderBlock (nn.Module):
    def __init__(self, in_c, out_c, time_emb_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = ConvBlock(out_c + out_c, out_c, time_emb_dim)

    def forward(self, inputs, skip, t):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x, t)
        return x

'''
Each Convolutional Block implements a sequence of convolutional layers, batch normalizations and relu activations and adds the time embedding as well.
''' 
class ConvBlock (nn.Module):
    def __init__(self, in_c, out_c, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_c)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, inputs, t):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        time_emb = self.time_mlp(t)
        time_emb = time_emb[(...,) + (None,) * 2]
        x = x + time_emb
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class Unet (nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        time_emb_dim = 32

       '''time embeddings'''
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        '''encoder'''
        self.e1 = EncoderBlock(in_channels, 64, time_emb_dim)
        self.e2 = EncoderBlock(64, 128, time_emb_dim)
        self.e3 = EncoderBlock(128, 256, time_emb_dim)
        self.e4 = EncoderBlock(256, 512, time_emb_dim)
        '''bridge'''
        self.b = ConvBlock(512, 1024, time_emb_dim)
       '''decoder'''
        self.d1 = DecoderBlock(1024, 512, time_emb_dim)
        self.d2 = DecoderBlock(512, 256, time_emb_dim)
        self.d3 = DecoderBlock(256, 128, time_emb_dim)
        self.d4 = DecoderBlock(128, 64, time_emb_dim)

        self.outputs = nn.Conv2d(64, out_channels, kernel_size=1, padding=0)

    def forward(self, inputs, timestep):
        '''encoder'''
        t = self.time_mlp(timestep)
        s1, p1 = self.e1(inputs, t)
        s2, p2 = self.e2(p1, t)
        s3, p3 = self.e3(p2, t)
        s4, p4 = self.e4(p3, t)
        '''bridge'''
        b = self.b(p4, t)
        '''decoder'''
        d1 = self.d1(b, s4, t)
        d2 = self.d2(d1, s3, t)
        d3 = self.d3(d2, s2, t)
        d4 = self.d4(d3, s1, t)

        outputs = self.outputs(d4)
        return outputs
