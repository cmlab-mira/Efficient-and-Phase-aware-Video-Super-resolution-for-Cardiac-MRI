import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import interpolate

from src.model.nets.base_net import BaseNet


class FRVSRNet(BaseNet):
    """The implementation of Frame-Recurrent Video Super-Resolution.
    
    ref:
        https://arxiv.org/pdf/1801.04590v4.pdf
        https://github.com/LoSealL/VideoSuperResolution
    
    Args:
        in_channels (int): The input channels.
        out_channels (int): The output channels.
        upscale_factor (int): The upscale factor.
        training (bool): The network for training or not. Default: True.
        num_resblocks (int): The number of the resblocks. Default: 10.
    """
    def __init__(self, in_channels, out_channels, upscale_factor, is_prediction=False, num_resblocks=10):
        super(FRVSRNet, self).__init__()
        self.upscale_factor = upscale_factor
        self.is_prediction = is_prediction
        self.srnet = SRNet(in_channels, out_channels, upscale_factor, num_resblocks)
        self.fnet = FNet(in_channels, 2)
        self.spatio_to_depth = SpaceToDepth(upscale_factor)
        self.warp = STN(mode='bilinear', padding_mode='border')
        
        # Xavier weights initialized
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.xavier_uniform_(m.weight)
        
    def forward(self, inputs):
        sr_imgs, lr_imgs = [], []
        n, c, h, w = inputs[0].shape
        lr_last = inputs[0]
        sr_last = torch.zeros(n, c, h*self.upscale_factor, w*self.upscale_factor, dtype=torch.float, device=inputs[0].device)
        for input in inputs:
            # SRNet part
            lr_flow = self.fnet(lr_last, input)
            sr_flow = interpolate(lr_flow, scale_factor=self.upscale_factor, mode='bilinear', align_corners=True)
            sr_img = self.spatio_to_depth(self.warp(sr_last.detach(), sr_flow[:, 0], sr_flow[:, 1]))
            sr_img = self.srnet(sr_img, input)
            sr_imgs.append(sr_img)
            sr_last = sr_img

            # FNet part
            lr_img = self.warp(lr_last, lr_flow[:, 0], lr_flow[:, 1])
            lr_imgs.append(lr_img)
            lr_last = input
        
        if self.is_prediction == False:
            return sr_imgs, lr_imgs
        else:
            return sr_imgs


class SRNet(nn.Module):
    """
    Args:
        in_channels (int): The input channels.
        out_channels (int): The output channels.
        upscale_factor (int): The upscale factor.
        num_resblocks (int): The number of the resblocks. Default: 10.
    """
    def __init__(self, in_channels, out_channels, upscale_factor, num_resblocks=10):
        super(SRNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upscale_factor = upscale_factor
        
        self.head = nn.Sequential()
        self.head.add_module('conv', nn.Conv2d(in_channels * (upscale_factor ** 2 + 1), 64, kernel_size=3, padding=1))
        self.head.add_module('relu', nn.ReLU(True))
        self.body = nn.Sequential(*[_ResBlock(64) for _ in range(num_resblocks)])
        self.tail = nn.Sequential()
        self.tail.add_module('deconv1', nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.tail.add_module('relu1', nn.ReLU(True))
        self.tail.add_module('deconv2', nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.tail.add_module('relu2', nn.ReLU(True))
        self.tail.add_module('conv', nn.Conv2d(64, out_channels, kernel_size=3, padding=1))

    def forward(self, *inputs):
        x = torch.cat(inputs, dim=1)
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x

    
class _ResBlock(nn.Module):
    def __init__(self, num_features):
        super(_ResBlock, self).__init__()
        self.body = nn.Sequential()
        self.body.add_module('conv1', nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
        self.body.add_module('relu1', nn.ReLU(True))
        self.body.add_module('conv2', nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))

    def forward(self, x):
        return x + self.body(x)


class FNet(nn.Module):
    """
    Args:
        in_channels (int): The input channels.
        out_channels (int): The output channels.
    """
    def __init__(self, in_channels, out_channels):
        super(FNet, self).__init__()
        f = 32
        
        self.body = nn.Sequential()
        in_features = in_channels * 2
        for i in range(3):
            self.body.add_module(f'conv{i+1}_1', nn.Conv2d(in_features, f, kernel_size=3, padding=1))
            self.body.add_module(f'leaky_relu{i+1}_1', nn.LeakyReLU(0.2, inplace=True))
            self.body.add_module(f'conv{i+1}_2', nn.Conv2d(f, f, kernel_size=3, padding=1))
            self.body.add_module(f'leaky_relu{i+1}_2', nn.LeakyReLU(0.2, inplace=True))
            self.body.add_module(f'pooling_{i}', nn.MaxPool2d(2))
            in_features = f
            f *= 2
        for i in range(3):
            self.body.add_module(f'conv{i+4}_1', nn.Conv2d(in_features, f, kernel_size=3, padding=1))
            self.body.add_module(f'leaky_relu{i+4}_1', nn.LeakyReLU(0.2, inplace=True))
            self.body.add_module(f'conv{i+4}_2', nn.Conv2d(f, f, kernel_size=3, padding=1))
            self.body.add_module(f'leaky_relu{i+4}_2', nn.LeakyReLU(0.2, inplace=True))
            self.body.add_module(f'upsample_{i}', BilinerUp(2))
            in_features = f
            f = f // 2
            
        self.tail = nn.Sequential()
        self.tail.add_module('conv1', nn.Conv2d(in_features, 32, kernel_size=3, padding=1))
        self.tail.add_module('leaky_relu', nn.LeakyReLU(0.2, inplace=True))
        self.tail.add_module('conv2', nn.Conv2d(32, out_channels, kernel_size=3, padding=1))
        self.tail.add_module('tanh', nn.Tanh())

    def forward(self, *inputs):
        x = torch.cat(inputs, dim=1)
        N, C, H, W = x.shape
        
        has_padded = False
        if H % 8 != 0 or W % 8 != 0:
            has_padded = True
            H_diff = 8 - H % 8 if H % 8 != 0 else 0
            W_diff = 8 - W % 8 if W % 8 != 0 else 0
            pad = (W_diff // 2, W_diff - W_diff//2, 
                   H_diff // 2, H_diff - H_diff//2)
            x = F.pad(x, pad, value=x.min())

        x = self.body(x)
        x = self.tail(x)
        
        if has_padded:            
            W0, Wn, H0, Hn = pad
            W0, Wn, H0, Hn = W0, x.size(-1) - Wn, H0, x.size(-2) - Hn
            x = x[..., H0:Hn, W0:Wn]
        
        return x
    
    
class BilinerUp(nn.Module):
    def __init__(self, scale_factor):
        super(BilinerUp, self).__init__()
        self.scale = scale_factor

    def forward(self, x):
        return interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
    

class SpaceToDepth(nn.Module):
    def __init__(self, downscale_factor):
        super(SpaceToDepth, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input):
        n, c, h, w = input.shape
        r = self.downscale_factor
        
        out_c = c * (r**2)
        out_h = h // r
        out_w = w // r

        input_view = input.contiguous().view(n, c, out_h, r, out_w, r)
        output = input_view.permute(0, 1, 3, 5, 2, 4).contiguous().view(n, out_c, out_h, out_w)
        return output
    
    
class STN(nn.Module):
    """Spatial transformer network. For optical flow based frame warping.
    Args:
        mode (str): Sampling interpolation mode of `grid_sample`.
        padding_mode (str): `zeros` | `borders`.
        normalize (bool): Flow value is normalized to [-1, 1] or absolute value.
    """
    def __init__(self, mode='bilinear', padding_mode='zeros', normalize=False):
        super(STN, self).__init__()
        self.mode = mode
        self.padding_mode = padding_mode
        self.norm = normalize

    def forward(self, inputs, u, v):
        batch = inputs.size(0)
        device = inputs.device
        mesh = self.nd_meshgrid(*inputs.shape[-2:], permute=[1, 0])
        mesh = torch.tensor(mesh, dtype=torch.float32, device=device)
        mesh = mesh.unsqueeze(0).repeat_interleave(batch, dim=0)
        
        # add flow to mesh
        _u, _v = u, v
        if self.norm:
            # flow needs to normalize to [-1, 1]
            h, w = inputs.shape[-2:]
            _u = u / w * 2
            _v = v / h * 2
        flow = torch.stack([_u, _v], dim=-1)
        assert flow.shape == mesh.shape
        mesh = mesh + flow
        return F.grid_sample(inputs, mesh, mode=self.mode, padding_mode=self.padding_mode)
    
    def nd_meshgrid(self, *size, permute=None):
        _error_msg = ("Permute index must match mesh dimensions, "
                      "should have {} indexes but got {}")
        size = np.array(size)
        ranges = []
        for x in size:
            ranges.append(np.linspace(-1, 1, x))
        mesh = np.stack(np.meshgrid(*ranges, indexing='ij'))
        if permute is not None:
            if len(permute) != len(size):
                raise ValueError(_error_msg.format(len(size), len(permute)))
            mesh = mesh[permute]
        return mesh.transpose(*range(1, mesh.ndim), 0)