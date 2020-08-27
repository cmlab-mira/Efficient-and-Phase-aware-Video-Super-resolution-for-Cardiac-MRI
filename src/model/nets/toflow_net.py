import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.nets.base_net import BaseNet


class TOFlowNet(BaseNet):
    """The implementation of the Task-Oriented Flow (TOFlow).

    Ref: https://arxiv.org/abs/1711.09078,
         https://github.com/open-mmlab/mmsr.
    """
    def __init__(self, in_channels, out_channels, num_frames, upscale_factor):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_frames = num_frames
        self.upscale_factor = upscale_factor
        self.ref_idx = num_frames // 2 if num_frames % 2 == 1 else num_frames // 2 - 1

        self.spy_net = SpyNet(2 * in_channels + 2)
        self.out_block = nn.Sequential(nn.Conv2d(in_channels * num_frames, 64, 9, 1, 4),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(64, 64, 9, 1, 4),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(64, 64, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(64, out_channels, 1))

    def forward(self, inputs):
        inputs = [F.interpolate(input, scale_factor=self.upscale_factor,
                                mode='bicubic', align_corners=False)
                  for input in inputs]
        x = torch.stack(inputs, dim=1)
        B, T, C, H, W = x.size()

        has_padded = False
        if H % 16 != 0 or W % 16 != 0:
            has_padded = True
            H_diff = 16 - H % 16 if H % 16 != 0 else 0
            W_diff = 16 - W % 16 if W % 16 != 0 else 0
            pad = (W_diff // 2, W_diff - W_diff//2,
                   H_diff // 2, H_diff - H_diff//2)
            x = F.pad(x, pad, value=x.min())
            B, T, C, H, W = x.size()

        x_ref = x[:, self.ref_idx, :, :, :]

        x_warped = []
        for i in range(self.num_frames):
            if i == self.ref_idx:
                x_warped.append(x_ref)
            else:
                x_nbr = x[:, i, :, :, :]
                flow = self.spy_net(x_ref, x_nbr).permute(0, 2, 3, 1)
                x_warped.append(flow_warp(x_nbr, flow))
        x_warped = torch.stack(x_warped, dim=1)

        x = x_warped.view(B, -1, H, W)
        output = self.out_block(x) + x_ref

        if has_padded:
            W0, Wn, H0, Hn = pad
            W0, Wn, H0, Hn = W0, output.size(-1) - Wn, H0, output.size(-2) - Hn
            output = output[..., H0:Hn, W0:Wn]
        return output


class SpyNet(nn.Module):
    """SpyNet for estimating optical flow
    Ranjan et al., Optical Flow Estimation using a Spatial Pyramid Network, 2016
    """
    def __init__(self, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList([SpyNet_Block(in_channels) for _ in range(4)])

    def forward(self, ref, nbr):
        B, C, H, W = ref.size()
        refs = [ref]
        nbrs = [nbr]
        for _ in range(3):
            refs.insert(0, F.avg_pool2d(refs[0], kernel_size=2, stride=2, count_include_pad=False))
            nbrs.insert(0, F.avg_pool2d(nbrs[0], kernel_size=2, stride=2, count_include_pad=False))

        flow = torch.zeros((B, 2, H // 16, W // 16), dtype=torch.float, device=ref.device)
        for i in range(4):
            flow_up = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0
            flow = flow_up + self.blocks[i](torch.cat([refs[i],
                                                       flow_warp(nbrs[i], flow_up.permute(0, 2, 3, 1)),
                                                       flow_up], dim=1))
        return flow


class SpyNet_Block(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels, 32, kernel_size=7, stride=1, padding=3),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64, 32, kernel_size=7, stride=1, padding=3),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(32, 16, kernel_size=7, stride=1, padding=3),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(16, 2, kernel_size=7, stride=1, padding=3))

    def forward(self, x):
        return self.block(x)


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, H, W, 2), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'

    Returns:
        Tensor: warped image or feature map
    """
    assert x.size()[-2:] == flow.size()[1:3]
    B, C, H, W = x.size()
    # mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float().to(x.device)  # W(x), H(y), 2
    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode)
    return output
