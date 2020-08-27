import torch
import torch.nn as nn
import math

from src.model.nets.base_net import BaseNet


class DRFNet(BaseNet):
    """The implementation of the Deep Recurrent Feedback Network (DRFN) for the Video Super-Resolution.

    The architecture is mainly inspired by the Super-Resolution FeedBack Network (SRFBN) and has some modification.
    First, it's for the Video Super-Resolution.
    Second, the global residual skip connection concatenates the features before and after the feedback block.
    Last, the simple deconvolution is replaced by the PixelShuffle module as used in the EDSR (ref: https://arxiv.org/pdf/1707.02921.pdf).

    Args:
        in_channels (int): The input channels.
        out_channels (int): The output channels.
        num_features (int): The number of the internel feature maps.
        num_groups (int): The number of the projection groups in the feedback block.
        upscale_factor (int): The upscale factor (2, 3, 4 or 8).
    """
    def __init__(self, in_channels, out_channels, num_features, num_groups, upscale_factor):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_features = num_features
        self.num_groups = num_groups

        if upscale_factor not in [2, 3, 4, 8]:
            raise ValueError(f'The upscale factor should be 2, 3, 4 or 8. Got {upscale_factor}.')
        self.upscale_factor = upscale_factor

        self.in_block = _InBlock(in_channels, num_features) # The input block.
        self.f_block = _FBlock(num_features, num_groups, upscale_factor) # The feedback block.
        self.out_block = _OutBlock(num_features, out_channels, upscale_factor) # The output block.

    def forward(self, inputs):
        outputs = []
        for i, input in enumerate(inputs):
            in_features = self.in_block(input)
            if i == 0:
                self.f_block.hidden_state = in_features # Reset the hidden state of the feedback block.
            f_features = self.f_block(in_features)
            self.f_block.hidden_state = f_features # Set the hidden state of the feedback block to the current feedback block output.
            features = in_features + f_features # The global residual skip connection.
            output = self.out_block(features)
            outputs.append(output)
        return outputs


class _InBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.add_module('conv1', nn.Conv2d(in_channels, 4 * out_channels, kernel_size=3, padding=1))
        self.add_module('prelu1', nn.PReLU(num_parameters=1, init=0.2))
        self.add_module('conv2', nn.Conv2d(4 * out_channels, out_channels, kernel_size=1))
        self.add_module('prelu2', nn.PReLU(num_parameters=1, init=0.2))


class _FBlock(nn.Module):
    def __init__(self, num_features, num_groups, upscale_factor):
        super().__init__()
        self.in_block = nn.Sequential()
        self.in_block.add_module('conv', nn.Conv2d(num_features * 2, num_features, kernel_size=1))
        self.in_block.add_module('prelu', nn.PReLU(num_parameters=1, init=0.2))

        self.up_blocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        if upscale_factor == 2:
            kernel_size, stride, padding = 6, 2, 2
        elif upscale_factor == 3:
            kernel_size, stride, padding = 7, 3, 2
        elif upscale_factor == 4:
            kernel_size, stride, padding = 8, 4, 2
        elif upscale_factor == 8:
            kernel_size, stride, padding = 12, 8, 2
        for i in range(num_groups):
            if i == 0:
                up_block = nn.Sequential()
                up_block.add_module('deconv', nn.ConvTranspose2d(num_features, num_features, kernel_size=kernel_size, stride=stride, padding=padding))
                up_block.add_module('prelu', nn.PReLU(num_parameters=1, init=0.2))
                self.up_blocks.append(up_block)

                down_block = nn.Sequential()
                down_block.add_module('conv', nn.Conv2d(num_features, num_features, kernel_size=kernel_size, stride=stride, padding=padding))
                down_block.add_module('prelu', nn.PReLU(num_parameters=1, init=0.2))
                self.down_blocks.append(down_block)
            else:
                up_block = nn.Sequential()
                up_block.add_module('conv1', nn.Conv2d(num_features * (i + 1), num_features, kernel_size=1))
                up_block.add_module('prelu1', nn.PReLU(num_parameters=1, init=0.2))
                up_block.add_module('deconv2', nn.ConvTranspose2d(num_features, num_features, kernel_size=kernel_size, stride=stride, padding=padding))
                up_block.add_module('prelu2', nn.PReLU(num_parameters=1, init=0.2))
                self.up_blocks.append(up_block)

                down_block = nn.Sequential()
                down_block.add_module('conv1', nn.Conv2d(num_features * (i + 1), num_features, kernel_size=1))
                down_block.add_module('prelu1', nn.PReLU(num_parameters=1, init=0.2))
                down_block.add_module('conv2', nn.Conv2d(num_features, num_features, kernel_size=kernel_size, stride=stride, padding=padding))
                down_block.add_module('prelu2', nn.PReLU(num_parameters=1, init=0.2))
                self.down_blocks.append(down_block)

        self.out_block = nn.Sequential()
        self.out_block.add_module('conv', nn.Conv2d(num_features * num_groups, num_features, kernel_size=1))
        self.out_block.add_module('prelu', nn.PReLU(num_parameters=1, init=0.2))

        self._hidden_state = None

    @property
    def hidden_state(self):
        return self._hidden_state

    @hidden_state.setter
    def hidden_state(self, state):
        self._hidden_state = state

    def forward(self, input):
        features = torch.cat([input, self.hidden_state], dim=1)
        lr_features = self.in_block(features)

        lr_features_list, hr_features_list = [lr_features], []
        for up_block, down_block in zip(self.up_blocks, self.down_blocks):
            concat_lr_features = torch.cat(lr_features_list, dim=1)
            hr_features = up_block(concat_lr_features)
            hr_features_list.append(hr_features)
            concat_hr_features = torch.cat(hr_features_list, dim=1)
            lr_features = down_block(concat_hr_features)
            lr_features_list.append(lr_features)

        features = torch.cat(lr_features_list[1:], dim=1)
        output = self.out_block(features)
        return output


class _OutBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, upscale_factor):
        super().__init__()
        if (math.log(upscale_factor, 2) % 1) == 0:
            for i in range(int(math.log(upscale_factor, 2))):
                self.add_module(f'conv{i+1}', nn.Conv2d(in_channels, 4 * in_channels, kernel_size=3, padding=1))
                self.add_module(f'pixelshuffle{i+1}', nn.PixelShuffle(2))
            self.add_module(f'conv{i+2}', nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        elif upscale_factor == 3:
            self.add_module('conv1', nn.Conv2d(in_channels, 9 * in_channels, kernel_size=3, padding=1))
            self.add_module('pixelshuffle1', nn.PixelShuffle(3))
            self.add_module('conv2', nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
