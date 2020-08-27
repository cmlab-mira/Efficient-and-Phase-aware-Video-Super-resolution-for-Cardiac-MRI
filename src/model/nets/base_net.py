import torch
import torch.nn as nn


class BaseNet(nn.Module):
    """The base class for all nets.
    """
    def __init__(self):
        super().__init__()

    def __repr__(self):
        params_size = sum([param.numel() for param in self.parameters() if param.requires_grad])
        return super().__repr__() + f'\nTrainable parameters: {params_size / 1e6} M\nMemory usage: {(params_size * 4) / (1 << 20)} MB'
