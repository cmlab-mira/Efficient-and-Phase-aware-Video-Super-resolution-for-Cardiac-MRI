import math
import torch
import torch.nn as nn

from src.model.nets.base_net import BaseNet


class Bicubic(BaseNet):
    """The implementation of the bicubic interpolation upscaling. 
    Args:
        upscale_factor (int): the upscale factor
    """
    def __init__(self, upscale_factor):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=upscale_factor, mode='bicubic', align_corners=True)
        
    def forward(self, input):
        output = self.upsample(input)
        return output