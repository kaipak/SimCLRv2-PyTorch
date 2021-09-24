import torch.nn as nn
import torch
from simclr.modules.resnet_v2 import get_resnet_v2
from simclr.modules.identity import Identity


class SimCLR(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to
    obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average
    pooling layer.
    """

    def __init__(self, encoder, projection_dim, n_features):
        super(SimCLR, self).__init__()

        self.encoder = encoder
        self.n_features = n_features

        # Replace the fc layer with an Identity function
        self.encoder.fc = Identity()

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i)
        # where σ is a ReLU non-linearity.
        # For v2 we have an added layer
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
        )

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        return h_i, h_j, z_i, z_j


class SimCLRv2(nn.Module):
    def __init__(self, resnet_depth: int = 50, resnet_width_multiplier: int = 2,
                 sk_ratio: float = 0.0625, pretrained_weights: str = None):
        """SimCLRv2 Implementation
            Using ResNet architecture from Pytorch converter which includes projection head.

        """
        super(SimCLRv2, self).__init__()
        self.encoder, self.projector = get_resnet_v2(depth=resnet_depth,
                                                     width_multiplier=resnet_width_multiplier,
                                                     sk_ratio=sk_ratio)
        if pretrained_weights:
            self.encoder.load_state_dict(torch.load(pretrained_weights, map_location='cpu')['resnet'])

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)
        z_i = self.projector(h_i)
        z_j = self.projector(h_j)

        return h_i, h_j, z_i, z_j
