import math
import torch
import torch.nn as nn

from src.model.nets.base_net import BaseNet


class EDSRNet(BaseNet):
    """The implementation of Enhanced Deep Residual Networks (ref: https://arxiv.org/pdf/1707.02921.pdf).

    Args:
        in_channels (int): The input channels.
        out_channels (int): The output channels.
        num_resblocks (int): The number of the resblocks.
        num_features (int): The number of the internel feature maps.
        upscale_factor (int): The upscale factor (2, 3 ,4 or 8).
        res_scale (float): The residual scaling factor of the resblocks. Default: `0.1`.
    """
    def __init__(self, in_channels, out_channels, num_resblocks, num_features, upscale_factor, res_scale=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_resblocks = num_resblocks
        self.num_features = num_features
        self.upscale_factor = upscale_factor
        self.res_scale = res_scale

        self.head = nn.Sequential(nn.Conv2d(self.in_channels, self.num_features, kernel_size=3, padding=1))
        self.body = nn.Sequential(*[_ResBlock(self.num_features, self.res_scale) for _ in range(self.num_resblocks)])
        self.body.add_module('conv', nn.Conv2d(self.num_features, self.num_features, kernel_size=3, padding=1))
        self.tail = nn.Sequential(_UpBlock(self.num_features, self.upscale_factor))
        self.tail.add_module('conv', nn.Conv2d(self.num_features, self.out_channels, kernel_size=3, padding=1))

    def forward(self, input):
        head = self.head(input)
        body = self.body(head) + head
        output = self.tail(body)
        return output


class _ResBlock(nn.Module):
    def __init__(self, num_features, res_scale):
        super().__init__()
        self.body = nn.Sequential()
        self.body.add_module('conv1', nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
        self.body.add_module('relu1', nn.ReLU())
        self.body.add_module('conv2', nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class _UpBlock(nn.Sequential):
    def __init__(self, num_features, upscale_factor):
        super().__init__()
        if (math.log(upscale_factor, 2) % 1) == 0:
            for i in range(int(math.log(upscale_factor, 2))):
                self.add_module(f'conv{i+1}', nn.Conv2d(num_features, 4 * num_features, kernel_size=3, padding=1))
                self.add_module(f'deconv{i+1}', nn.PixelShuffle(2))
        elif upscale_factor == 3:
            self.add_module(f'conv1', nn.Conv2d(num_features, 9 * num_features, kernel_size=3, padding=1))
            self.add_module(f'deconv1', nn.PixelShuffle(3))
        else:
            raise NotImplementedError
