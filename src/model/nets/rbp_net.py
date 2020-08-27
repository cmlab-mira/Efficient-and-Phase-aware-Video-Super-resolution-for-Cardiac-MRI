import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.nets.base_net import BaseNet


class RBPNet(BaseNet):
    """The implementation of the Recurrent Back-Projection Network (RBPN).

    Ref: https://arxiv.org/abs/1903.10128,
         https://github.com/alterzero/RBPN-PyTorch.
    """
    def __init__(self, in_channels, out_channels, base_filter, feat, num_stages, num_resblocks, num_frames, upscale_factor):
        super().__init__()
        self.t = num_frames // 2 if num_frames % 2 == 1 else num_frames // 2 - 1

        if upscale_factor == 2:
            kernel_size, stride, padding = 6, 2, 2
        elif upscale_factor == 3:
            kernel_size, stride, padding = 7, 3, 2
        elif upscale_factor == 4:
            kernel_size, stride, padding = 8, 4, 2
        elif upscale_factor == 8:
            kernel_size, stride, padding = 12, 8, 2

        #Initial Feature Extraction
        self.feat0 = ConvBlock(in_channels, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.feat1 = ConvBlock(in_channels * 2, base_filter, 3, 1, 1, activation='prelu', norm=None)

        ###DBPNS
        self.dbp_net = DBPNet(base_filter, feat, num_stages, upscale_factor)

        #Res-Block1
        modules_body1 = [ResnetBlock(base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None)
                         for _ in range(num_resblocks)]
        modules_body1.append(DeconvBlock(base_filter, feat, kernel_size, stride, padding, activation='prelu', norm=None))
        self.res_feat1 = nn.Sequential(*modules_body1)

        #Res-Block2
        modules_body2 = [ResnetBlock(feat, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None)
                         for _ in range(num_resblocks)]
        modules_body2.append(ConvBlock(feat, feat, 3, 1, 1, activation='prelu', norm=None))
        self.res_feat2 = nn.Sequential(*modules_body2)

        #Res-Block3
        modules_body3 = [ResnetBlock(feat, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None)
                         for _ in range(num_resblocks)]
        modules_body3.append(ConvBlock(feat, base_filter, kernel_size, stride, padding, activation='prelu', norm=None))
        self.res_feat3 = nn.Sequential(*modules_body3)

        #Reconstruction
        self.output = ConvBlock((num_frames - 1) * feat, out_channels, 3, 1, 1, activation=None, norm=None)
        """
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
        """
    def forward(self, inputs):
        x = inputs.pop(self.t)
        neighbors = inputs

        ### initial feature extraction
        feat_input = self.feat0(x)
        feat_frame = []
        for j in range(len(neighbors)):
            feat_frame.append(self.feat1(torch.cat([x, neighbors[j]], dim=1)))

        ### Projection
        Ht = []
        for j in range(len(neighbors)):
            h0 = self.dbp_net(feat_input)
            h1 = self.res_feat1(feat_frame[j])

            e = h0 - h1
            e = self.res_feat2(e)
            h = h0 + e
            Ht.append(h)
            feat_input = self.res_feat3(h)

        ####Reconstruction
        out = torch.cat(Ht, dim=1)
        output = self.output(out)
        return output


class DBPNet(nn.Module):
    def __init__(self, base_filter, feat, num_stages, upscale_factor):
        super().__init__()
        if upscale_factor == 2:
            kernel_size, stride, padding = 6, 2, 2
        elif upscale_factor == 3:
            kernel_size, stride, padding = 7, 3, 2
        elif upscale_factor == 4:
            kernel_size, stride, padding = 8, 4, 2
        elif upscale_factor == 8:
            kernel_size, stride, padding = 12, 8, 2

        #Initial Feature Extraction
        #self.feat0 = ConvBlock(num_channels, feat, 3, 1, 1, activation='prelu', norm=None)
        self.feat1 = ConvBlock(base_filter, feat, 1, 1, 0, activation='prelu', norm=None)
        #Back-projection stages
        self.up1 = UpBlock(feat, kernel_size, stride, padding)
        self.down1 = DownBlock(feat, kernel_size, stride, padding)
        self.up2 = UpBlock(feat, kernel_size, stride, padding)
        self.down2 = DownBlock(feat, kernel_size, stride, padding)
        self.up3 = UpBlock(feat, kernel_size, stride, padding)
        #Reconstruction
        self.output = ConvBlock(num_stages*feat, feat, 1, 1, 0, activation=None, norm=None)
        """
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
        """
    def forward(self, x):
        #x = self.feat0(x)
        x = self.feat1(x)

        h1 = self.up1(x)
        h2 = self.up2(self.down1(h1))
        h3 = self.up3(self.down2(h2))

        x = self.output(torch.cat((h3, h2, h1),1))

        return x


class ConvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None):
        super().__init__()
        self.conv = nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm=None):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class ResnetBlock(nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm='batch'):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)
        self.conv2 = nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = nn.BatchNorm2d(num_filter)
        elif norm == 'instance':
            self.bn = nn.InstanceNorm2d(num_filter)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()


    def forward(self, x):
        residual = x
        if self.norm is not None:
            out = self.bn(self.conv1(x))
        else:
            out = self.conv1(x)

        if self.activation is not None:
            out = self.act(out)

        if self.norm is not None:
            out = self.bn(self.conv2(out))
        else:
            out = self.conv2(out)

        out = torch.add(out, residual)

        if self.activation is not None:
            out = self.act(out)

        return out


class UpBlock(nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super().__init__()
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class DownBlock(nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super().__init__()
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0
