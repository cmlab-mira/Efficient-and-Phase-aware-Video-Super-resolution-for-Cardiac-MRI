import math
import torch
import torch.nn as nn
import numpy as np

from src.model.nets.base_net import BaseNet


class DUFNet(BaseNet):
    """The implementation of the Deep Video Super-Resolution Network Using Dynamic Upsampling Filters Without Explicit Motion Compensation
    ref: http://openaccess.thecvf.com/content_cvpr_2018/papers/Jo_Deep_Video_Super-Resolution_CVPR_2018_paper.pdf

    Args:
        in_channels (int): The input channels.
        out_channels (int): The output channels.
        num_frames (int): The number of input frames.
        size_filter (int): The kernel size in the DUF component.
        upscale_factor (int): The upscale factor (2, 3, 4).
        backbone (str): The denseNet backbone used in the network.
    """
    def __init__(self, in_channels, out_channels, num_frames, size_filter, upscale_factor, backbone):
        super().__init__()
        self.num_frames = num_frames
        self.size_filter = size_filter
        self.upscale_factor = upscale_factor

        assert backbone in ['_DenseLayer16', '_DenseLayer28', '_DenseLayer52']
        if backbone == '_DenseLayer16':
            self.denseLayer = _DenseLayer16(64, 32)
        elif backbone == '_DenseLayer28':
            self.denseLayer = _DenseLayer28(64, 16)
        elif backbone == '_DenseLayer52':
            self.denseLayer = _DenseLayer52(64, 16)

        self.head = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)

        # The blocks in the end of the FilterGenerationNetwork
        self.filterNet = nn.Sequential()
        self.filterNet.add_module('relu1', nn.ReLU())
        self.filterNet.add_module('conv1', nn.Conv3d(256, 512, kernel_size=1))
        self.filterNet.add_module('relu2', nn.ReLU())
        self.filterNet.add_module('conv2', nn.Conv3d(512, (size_filter ** 2) * (upscale_factor ** 2), kernel_size=1))

        # The blocks in the end of the ResidualGenerationNetwork
        self.residualNet = nn.Sequential()
        self.residualNet.add_module('relu1', nn.ReLU())
        self.residualNet.add_module('conv1', nn.Conv3d(256, 256, kernel_size=1))
        self.residualNet.add_module('relu2', nn.ReLU())
        self.residualNet.add_module('conv2', nn.Conv3d(256, in_channels * (upscale_factor ** 2), kernel_size=1))

    def forward(self, inputs):
        # dim: (B, C, 1, H, W)
        t = self.num_frames // 2 if self.num_frames % 2 == 1 else self.num_frames // 2 - 1
        target = inputs[t].unsqueeze(dim=2)

        # head
        features = []
        for i in range(self.num_frames):
            feature = self.head(inputs[i])
            features.append(feature)
        features = torch.stack(features, dim=2)

        # denseLayer
        features = self.denseLayer(features)

        # filter generation network
        filters = self.filterNet(features)
        # dim: (N, size_filter^2, upscale_factor^2, D, H, W)
        filters = filters.reshape(filters.shape[0], (self.size_filter ** 2), (self.upscale_factor ** 2), *filters.shape[2:])
        filters = nn.functional.softmax(filters, dim=1)
        # dim: (N, size_filter^2, upscale_factor^2, H, W)
        filters = filters[:, :, :, 0, ...]

        outputs = []
        for c in range(target.shape[1]):
            # dim: (N, 1, H, W)
            x = target[:, c, ...]
            # dim: (size_filter^2, 1, H, W)
            filter_localexpand = np.reshape(np.eye(np.prod(self.size_filter ** 2), np.prod(self.size_filter ** 2)),
                                            (np.prod(self.size_filter ** 2), 1, self.size_filter, self.size_filter))
            filter_localexpand = torch.FloatTensor(filter_localexpand).to(x.device)
            x = nn.functional.conv2d(x, filter_localexpand, padding=self.size_filter//2)
            # dim: (N, H, W, 1, size_filter^2)
            x = x.permute(0, 2, 3, 1).contiguous().unsqueeze(dim=-2)
            # dim: (N, H, W, size_filter^2, upscale_factor^2)
            f = filters.permute(0, 3, 4, 1, 2).contiguous()
            # dim: (N, H, W, upscale_factor^2)
            x = torch.matmul(x, f).squeeze(dim=-2).permute(0, 3, 1, 2).contiguous()
            x = nn.functional.pixel_shuffle(x, self.upscale_factor)
            outputs.append(x)
        outputs = torch.cat(outputs, dim=1)

        # residual generation network
        residual = self.residualNet(features)
        residual = residual.squeeze(dim=2)
        residual = nn.functional.pixel_shuffle(residual, self.upscale_factor)
        outputs = outputs + residual

        return outputs


class _DenseLayer16(nn.Module):
    def __init__(self, F, G):
        super().__init__()

        # Dense Layer
        for i in range(0, 3):
            setattr(self, f'conv{i}', _denseBlock1(F, G))
            F += G
        for i in range(3, 6):
            setattr(self, f'conv{i}', _denseBlock2(F, G))
            F += G

        # Tail block
        self.tail = nn.Sequential()
        self.tail.add_module('bn', nn.BatchNorm3d(256))
        self.tail.add_module('relu', nn.ReLU())
        self.tail.add_module('conv', nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)))

    def forward(self, x):
        concat = x
        for i in range(0, 6):
            conv = getattr(self, f'conv{i}')
            x = conv(concat)
            if i >= 3:
                concat = torch.cat((concat[:, :, 1:-1], x), dim=1)
            else:
                concat = torch.cat((concat, x), dim=1)
        x = self.tail(concat)
        return x


class _DenseLayer28(nn.Module):
    def __init__(self, F, G):
        super().__init__()

        # Dense Layer
        for i in range(0, 9):
            setattr(self, f'conv{i}', _denseBlock1(F, G))
            F += G
        for i in range(9, 12):
            setattr(self, f'conv{i}', _denseBlock2(F, G))
            F += G

        # Tail block
        self.tail = nn.Sequential()
        self.tail.add_module('bn', nn.BatchNorm3d(256))
        self.tail.add_module('relu', nn.ReLU())
        self.tail.add_module('conv', nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)))

    def forward(self, x):
        concat = x
        for i in range(0, 12):
            conv = getattr(self, f'conv{i}')
            x = conv(concat)
            if i >= 9:
                concat = torch.cat((concat[:, :, 1:-1], x), dim=1)
            else:
                concat = torch.cat((concat, x), dim=1)
        x = self.tail(concat)
        return x


class _DenseLayer52(nn.Module):
    def __init__(self, F, G):
        super().__init__()

        # Dense Layer
        for i in range(0, 21):
            setattr(self, f'conv{i}', _denseBlock1(F, G))
            F += G
        for i in range(21, 24):
            setattr(self, f'conv{i}', _denseBlock2(F, G))
            F += G

        # Tail block
        self.tail = nn.Sequential()
        self.tail.add_module('bn', nn.BatchNorm3d(448))
        self.tail.add_module('relu', nn.ReLU())
        self.tail.add_module('conv', nn.Conv3d(448, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)))

    def forward(self, x):
        concat = x
        for i in range(0, 24):
            conv = getattr(self, f'conv{i}')
            x = conv(concat)
            if i >= 21:
                concat = torch.cat((concat[:, :, 1:-1], x), dim=1)
            else:
                concat = torch.cat((concat, x), dim=1)
        x = self.tail(concat)
        return x


class _denseBlock1(nn.Sequential):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.add_module('bn1', nn.BatchNorm3d(in_features))
        self.add_module('relu1', nn.ReLU())
        self.add_module('conv1', nn.Conv3d(in_features, in_features, kernel_size=1))
        self.add_module('bn2', nn.BatchNorm3d(in_features))
        self.add_module('relu2', nn.ReLU())
        self.add_module('conv2', nn.Conv3d(in_features, out_features, kernel_size=3, padding=1))


class _denseBlock2(nn.Sequential):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.add_module('bn1', nn.BatchNorm3d(in_features))
        self.add_module('relu1', nn.ReLU())
        self.add_module('conv1', nn.Conv3d(in_features, in_features, kernel_size=1))
        self.add_module('bn2', nn.BatchNorm3d(in_features))
        self.add_module('relu2', nn.ReLU())
        self.add_module('conv2', nn.Conv3d(in_features, out_features, kernel_size=3, padding=(0, 1, 1)))
