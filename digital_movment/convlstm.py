from typing import List

import torch.nn as nn
from torch.autograd import Variable
import torch


class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        
        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
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

        self.height, self.width = input_size
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias        = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

        torch.nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, input_tensor, cur_state):
        
        h_cur, c_cur = cur_state
        
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) 
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda(),
                Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda())


class ConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]

            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)
    # 是一个单向的LSTM网络
    def forward(self, input_tensor, hidden_state=None):
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
        # 图片应该是32，t， 1， 64， 64 之类
        input_tensor = input_tensor.unsqueeze(2).float()
        assert len(input_tensor.shape) == 5

        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list   = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):

                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1: ] # 取出最后一层
            last_state_list   = last_state_list[-1: ]

        assert isinstance(layer_output_list, list)
        assert isinstance(last_state_list, list)
        # return layer_output_list, last_state_list
        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

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


class ConvLSTMDeocder(ConvLSTM):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers, decode_step,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTMDeocder, self).__init__(input_size, input_dim, hidden_dim, kernel_size, num_layers,
                                              batch_first, bias, return_all_layers)
        self.decode_step = decode_step

    def forward(self, input_tensor: torch.Tensor, hidden_state: List):

        input_tensor = input_tensor.unsqueeze(1)
        assert len(hidden_state) == self.num_layers
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        assert hidden_state is not None
        assert input_tensor.size(1) == 1 # decode步骤

        layer_output_list = []
        last_state_list = []

        seq_len = self.decode_step
        cur_layer_input = input_tensor


        # 一步一步的解码
        for t in range(seq_len):
            output_inner = []  # 对于每一层而言
            cur_state = []
            for layer_idx in range(self.num_layers):
                h, c = hidden_state[layer_idx]
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, 0, :, :, :],  # 取上一步的输出结果作为输入
                                                 cur_state=[h, c])
                cur_layer_input = h.unsqueeze(1)

                output_inner.append(h)
                cur_state.append([h, c]) # 记录当前t时刻的每一层的状态
            hidden_state = cur_state

            layer_output = output_inner
            layer_output_list.append(layer_output[:])
            last_state_list.append(hidden_state[:])



        if not self.return_all_layers:
            layer_output_list = [w[-1] for w in layer_output_list] # 每一个时刻的最后一层
            last_state_list =   last_state_list[-1: ] # 对于状态而言，我们只关心最后一个时刻的隐层状态

        # return layer_output_list, last_state_list
        return layer_output_list, last_state_list

class EncoderDecoder(torch.nn.Module):
    def __init__(self, encoder, decoder, reverse = False):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.reverse = reverse
        # self.decode_step = decode_step

    def forward(self, input_tensor):
        layer_output_list, last_state_list = self.encoder(input_tensor)
        last_input = last_state_list[-1][-1] # 最后一层的最后一个输出作为decoder的输入 # torch.Size([32, 128, 64, 64])
        last_state = last_state_list # 每一层最后一个位置作为隐层, 似乎LSTM不会保存seq中间的隐层

        if self.reverse:
            last_state = last_state[::-1]

        output_list, state_list = self.decoder(last_input, last_state)

        ans = torch.cat(output_list, dim=1)
        assert ans.size(1) == self.decoder.decode_step
        assert isinstance(ans, torch.Tensor)
        return ans


class Encorder(torch.nn.Module):
    def __init__(self, encorder):
        super(Encorder, self).__init__()
        self.encoder = encorder

    def forward(self, input_tensor):
        layer_output_list, last_state_list = self.encoder(input_tensor)
        return layer_output_list[-1].squeeze(2)

from fastai.vision import *


def mysimple_cnn(actns: Collection[int], kernel_szs: Collection[int] = None,
                 strides: Collection[int] = None, bn=False) -> nn.Sequential:
    "CNN with `conv_layer` defined by `actns`, `kernel_szs` and `strides`, plus batchnorm if `bn`."
    nl = len(actns) - 1
    kernel_szs = ifnone(kernel_szs, [4] * nl)
    strides = ifnone(strides, [1] * nl)
    layers = [conv_layer(actns[i], actns[i + 1], kernel_szs[i], stride=strides[i],
                         norm_type=(NormType.Batch if bn and i < (len(strides) - 1) else None)) for i in
              range_of(strides)]
    return nn.Sequential(*layers)

def create_unet(body, size,  pretrained:bool=True, blur_final:bool=True,
                 norm_type:Optional[NormType]=NormType, split_on:Optional[SplitFuncOrIdxList]=None, blur:bool=False,
                 self_attention:bool=False, y_range:Optional[Tuple[float,float]]=None, last_cross:bool=True,
                 bottle:bool=False, cut:Union[int,Callable]=None, **learn_kwargs:Any):
    "Build Unet learner from `data` and `arch`."

    model = models.unet.DynamicUnet(body, n_classes=1, img_size=size, blur=blur, blur_final=blur_final,
          self_attention=self_attention, y_range=y_range, norm_type=norm_type, last_cross=last_cross,
          bottle=bottle)
    apply_init(model[2], nn.init.kaiming_normal_)
    return model

class UnetXConvLSTM(ConvLSTMCell):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        super(UnetXConvLSTM, self).__init__(input_size, input_dim, hidden_dim, kernel_size, bias)
        # self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
        #                       out_channels=4 * self.hidden_dim,
        #                       kernel_size=self.kernel_size,
        #                       padding=self.padding,
        #                       bias=self.bias)
        in_channels = self.input_dim + self.hidden_dim
        out_channels = 4 * self.hidden_dim
        model = mysimple_cnn([in_channels, 128, 64, out_channels])
        self.conv = create_unet(model, (64, 64) ) # TODO: remove this hard code

class UnetConvLSTM(ConvLSTM):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(UnetConvLSTM, self).__init__(input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False)
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(UnetXConvLSTM(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)