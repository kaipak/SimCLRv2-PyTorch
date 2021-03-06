import torch.nn as nn
import torch
from simclr.modules.resnet import get_resnet
from simclr.modules.identity import Identity


class SimCLRv2(nn.Module):
    """SimCLRv2 Implementation
        Using ResNet architecture from Pytorch converter which includes projection head.

    """
    def __init__(self, resnet_depth: int = 50, resnet_width_multiplier: int = 2,
                 sk_ratio: float = 0.0625, pretrained_weights: str = None):
        super(SimCLRv2, self).__init__()
        self.encoder, self.projector = get_resnet(depth=resnet_depth,
                                                     width_multiplier=resnet_width_multiplier,
                                                     sk_ratio=sk_ratio)
        if pretrained_weights:
            self.encoder.load_state_dict(torch.load(pretrained_weights, map_location='cpu')['resnet'])
            self.projector.load_state_dict(torch.load(pretrained_weights, map_location='cpu')['head'])

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)
        z_i = self.projector(h_i)
        z_j = self.projector(h_j)

        return h_i, h_j, z_i, z_j


class SimCLRv2_ft(nn.Module):
    """Take a pretrained SimCLRv2 Model and Finetune with linear layer"""
    def __init__(self, simclrv2_model, n_classes):
        super(SimCLRv2_ft, self).__init__()
        self.encoder = simclrv2_model.encoder
        # From v2 paper, we just need the first layer from projector
        self.projector = torch.nn.Sequential(*(list(simclrv2_model.projector.children())[0][:2]))
        # Hack
        linear_in_features = self.projector[0].out_features
        self.linear = nn.Linear(linear_in_features, n_classes)

    def forward(self, x):
        h = self.encoder(x)
        h_prime = self.projector(h)
        y_hat = self.linear(h_prime)

        return y_hat


