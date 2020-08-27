import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import numpy as np

from src.model.nets.base_net import BaseNet


class RefineNet(BaseNet):
    """
    Args:
        in_channels (int): The input channels.
        out_channels (int): The output channels.
        num_features (list of int): The number of the internel feature maps.
        upscale_factor (int): The upscale factor (2, 3, 4 or 8).
    """
    def __init__(self, in_channels, out_channels, num_features, num_stages=1, refine_window_size=5, upscale_factor=4, 
                 update_memory=False, num_updated_frames=0, memory=True, positional_encoding=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_features = num_features
        self.num_stages = num_stages
        self.refine_window_size = refine_window_size
        self.upscale_factor = upscale_factor
        self.update_memory = update_memory
        self.num_updated_frames = num_updated_frames
        
        if upscale_factor not in [2, 3, 4, 8]:
            raise ValueError(f'The upscale factor should be 2, 3, 4 or 8. Got {upscale_factor}.')
        
        if update_memory == False and num_updated_frames != 0:
            raise ValueError('The \"update_memory\" is not activated!')
        
        num_feature = num_features[0]
        self.in_block = _InBlock(in_channels, num_feature) # The input block.
        self.forward_lstm_block = _ConvLSTM(input_dim=num_feature,
                                            hidden_dim=num_features,
                                            kernel_size=(3, 3),
                                            num_layers=len(num_features),
                                            bias=True,
                                            memory=memory)
        self.backward_lstm_block = _ConvLSTM(input_dim=num_feature,
                                             hidden_dim=num_features,
                                             kernel_size=(3, 3),
                                             num_layers=len(num_features),
                                             bias=True,
                                             memory=memory)
        if positional_encoding:
            refine_in_features = refine_window_size * (num_features[-1] * 2 + 1)
        else:
            refine_in_features = refine_window_size * (num_features[-1] * 2)
        self.refine_block = _RefineBlock(refine_in_features, 
                                         num_features[-1], 
                                         refine_window_size, 
                                         num_updated_frames, 
                                         positional_encoding)
        self.out_block = _OutBlock(num_feature, out_channels, upscale_factor)
        
    def forward(self, inputs, pos_codes):
        in_features, forward_update_features, backward_update_features = [], [], []
        outputs = []
        num_frames = len(inputs) - 2 * self.num_updated_frames
        
        for input in inputs[self.num_updated_frames:-self.num_updated_frames]:
            in_features.append(self.in_block(input))
        
        for _ in range(self.num_stages):
            forward_h_t, backward_h_t = [], []
            self.forward_lstm_block._init_hidden(in_features[0].size(0), in_features[0].size(2), in_features[0].size(3))
            self.backward_lstm_block._init_hidden(in_features[0].size(0), in_features[0].size(2), in_features[0].size(3))
            
            with torch.no_grad():
                if len(forward_update_features) == 0:
                    for input in inputs[:self.num_updated_frames]:
                        forward_update_features.append(self.in_block(input))
                    for input in inputs[-self.num_updated_frames:]:
                        backward_update_features.append(self.in_block(input))
                    
            _features = forward_update_features + in_features + backward_update_features
            for i, feature in enumerate(_features):
                if (i >= self.num_updated_frames) and (i < (len(_features) - self.num_updated_frames)):
                    forward_h_t.append(self.forward_lstm_block(feature))
                else:
                    with torch.no_grad():
                        forward_h_t.append(self.forward_lstm_block(feature))
            for i, feature in enumerate(reversed(_features)):
                if (i >= self.num_updated_frames) and (i < (len(_features) - self.num_updated_frames)):
                    backward_h_t.insert(0, self.backward_lstm_block(feature))
                else:
                    with torch.no_grad():
                        backward_h_t.insert(0, self.backward_lstm_block(feature))
            refine_maps = self.refine_block(forward_h_t, backward_h_t, pos_codes)
            
            #######################################################################################################
            # Model Output
            #######################################################################################################
            # Forward
            _outputs = []
            for i in range(num_frames):
                _outputs.append(self.out_block(in_features[i] + forward_h_t[i+self.num_updated_frames]))
            outputs.append(_outputs)
            # Backward
            _outputs = []
            for i in range(num_frames):
                _outputs.append(self.out_block(in_features[i] + backward_h_t[i+self.num_updated_frames]))
            outputs.append(_outputs)
            # Fused
            _outputs = []
            for i in range(num_frames):
                _outputs.append(self.out_block(in_features[i] + refine_maps[i+self.num_updated_frames-self.refine_window_size//2]))
            outputs.append(_outputs)
            
            ########################################################################################################
            # Update the features after refined
            ########################################################################################################
            if self.num_stages > 1:
                # Forward
                for i in range(len(forward_update_features)):
                    if i < self.refine_window_size // 2:
                        forward_update_features[i] += forward_h_t[i]
                    else:
                        forward_update_features[i] += refine_maps[i-self.refine_window_size//2]
                # Backward
                for i in range(len(backward_update_features)):
                    if i < self.refine_window_size // 2:
                        backward_update_features[-i-1] += backward_h_t[-i-1]
                    else:
                        backward_update_features[-i-1] += refine_maps[-i+self.refine_window_size//2-1]
                # Center (Main)
                for i in range(len(in_features)):
                    in_features[i] += refine_maps[i+self.num_updated_frames-self.refine_window_size//2]
            
        return tuple(outputs)
    
    
class _RefineBlock(nn.Module):
    def __init__(self, in_channels, num_features, num_frames, num_updated_frames, positional_encoding=False):
        super().__init__()
        self.in_channels = in_channels
        self.num_features = num_features
        self.num_frames = num_frames
        self.num_updated_frames = num_updated_frames
        self.positional_encoding = positional_encoding
        
        self.body = nn.Sequential()
        if self.positional_encoding:
            self.body.add_module('conv1', nn.Conv2d(in_channels, in_channels // num_frames, kernel_size=3, padding=1))
            self.add_module('prelu', nn.PReLU(num_parameters=1, init=0.2))
            self.body.add_module('conv2', nn.Conv2d(in_channels // num_frames, num_features, kernel_size=3, padding=1))
            self.add_module('prelu', nn.PReLU(num_parameters=1, init=0.2))
        else:
            self.body.add_module('conv1', nn.Conv2d(in_channels, num_features, kernel_size=1))
            self.add_module('prelu', nn.PReLU(num_parameters=1, init=0.2))
        
    def forward(self, forward_features, backward_features, pos_codes):
        """
        Args:
            forward_features (list of FloatTensor): The forward hidden features.
            backward_features (list of FloatTensor): The backward hidden features.
            pos_codes (FloatTensor): The positional encoding.
        """
        N, C, H, W = forward_features[0].shape
        half_window_size = self.num_frames//2
        forward_features = torch.stack(forward_features, dim=1)
        backward_features = torch.stack(backward_features, dim=1)
        pos_codes = pos_codes.repeat(H, W, 1, 1, 1).permute(2, 3, 4, 0, 1).contiguous()
        if self.positional_encoding:
            features = torch.cat((forward_features, backward_features, pos_codes), dim=2)
        else:
            features = torch.cat((forward_features, backward_features), dim=2)
        
        refine_maps = []
        for i in range(half_window_size, features.shape[1] - half_window_size):
            feature = features[:, i-half_window_size:i+half_window_size+1]
            feature = torch.cat([feature[:, j] for j in range(feature.shape[1])], dim=1)
            
            if i >= self.num_updated_frames and i < (forward_features.shape[1] - self.num_updated_frames):
                refine_maps.append(self.body(feature))
            else:
                with torch.no_grad():
                    refine_maps.append(self.body(feature))
            
        return refine_maps
        

class _InBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        self.add_module('prelu', nn.PReLU(num_parameters=1, init=0.2))

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
    
    
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias, memory):
        """
        Initialize ConvLSTM cell.
        
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.memory = memory
        
        if memory:
            self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                  out_channels=4 * self.hidden_dim,
                                  kernel_size=self.kernel_size,
                                  padding=self.padding,
                                  bias=self.bias)
        else:
            self.conv = nn.Conv2d(in_channels=self.input_dim * 2,
                                  out_channels=4 * self.hidden_dim,
                                  kernel_size=self.kernel_size,
                                  padding=self.padding,
                                  bias=self.bias)

    def forward(self, input_tensor, cur_state):
        
        h_cur, c_cur = cur_state
        
        if self.memory:
            # concatenate along channel axis
            combined = torch.cat([input_tensor, h_cur], dim=1)
        else:
            combined = torch.cat([input_tensor, input_tensor], dim=1)
        
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size, height, width):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, height, width)).cuda(),
                Variable(torch.zeros(batch_size, self.hidden_dim, height, width)).cuda())


class _ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, bias=True, memory=True):
        super(_ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.bias = bias
        self.hidden_state = None

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias,
                                          memory=memory))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor,):
        """
        
        Parameters
        ----------
        input_tensor: todo 
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
            
        Returns
        -------
        last_state_list, layer_output
        """
        
        cur_layer_input = input_tensor
        for layer_idx in range(self.num_layers):
            h, c = self.hidden_state[layer_idx]
            h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input, cur_state=[h, c])
            self.hidden_state[layer_idx] = (h, c)
            cur_layer_input = h

        return h

    def _init_hidden(self, batch_size, height, width):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, height, width))
        self.hidden_state = init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param