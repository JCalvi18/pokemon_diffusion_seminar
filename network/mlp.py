from torch import nn


class MLP (nn.Module):
    def __init__ (self):
        super (MLP, self).__init__ ()
        self.layers = nn.Sequential (
            nn.Linear (784, 100),
            nn.ReLU (),
            nn.Linear (100, 10)
        )

    def forward (self, x):
        # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        x = x.view (x.size (0), -1)
        x = self.layers (x)
        return x