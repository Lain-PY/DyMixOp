import torch
import torch.nn as nn
import os
import numpy as np
from torch.autograd import Variable


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"
torch.manual_seed(0)
np.random.seed(0)


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias, device, padding, dim_mode=1):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = padding
        self.bias = bias
        self.device = device
        self.dim_mode = dim_mode

        if self.dim_mode == 1:
            self.conv = nn.Conv1d(in_channels=self.input_dim + self.hidden_dim,
                                  out_channels=4 * self.hidden_dim,
                                  kernel_size=self.kernel_size,
                                  padding=self.padding,
                                  bias=self.bias,
                                  dtype=torch.float32).to(device)
        elif self.dim_mode == 2:
            self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                  out_channels=4 * self.hidden_dim,
                                  kernel_size=self.kernel_size,
                                  padding=self.padding,
                                  bias=self.bias,
                                  dtype=torch.float32).to(device)
        elif self.dim_mode == 3:
            self.conv = nn.Conv3d(in_channels=self.input_dim + self.hidden_dim,
                                  out_channels=4 * self.hidden_dim,
                                  kernel_size=self.kernel_size,
                                  padding=self.padding,
                                  bias=self.bias,
                                  dtype=torch.float32).to(device)
        else:
            raise ValueError(f"Invalid dimension mode: {self.dim_mode}. Must be 1, 2, or 3.")

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

    def init_hidden(self, hidden_shape):
        # hidden_shape: (batch, hidden_dim, spatial_dims...)
        # We assume hidden_shape comes with the correct number of dimensions
        return (Variable(torch.zeros(hidden_shape).to(self.device)).type(torch.float32),
                Variable(torch.zeros(hidden_shape).to(self.device)).type(torch.float32))


class ConvLSTM(nn.Module):
    def __init__(self, model_config, device):
        super(ConvLSTM, self).__init__()

        self.seq_len = model_config.ar_nseq_train + model_config.ar_nseq_test
        self.channel_dim = model_config.input_dim
        self.rnn_input_dim = model_config.coor_input_dim
        self.rnn_hidden_dim = model_config.hidden_dim
        self.rnn_kernel_size = model_config.kernel_size
        self.rnn_output_dim = model_config.output_dim
        self.involve_history_step = model_config.inp_involve_history_step

        self.bias = True
        self.return_all_layers = False
        self.num_layers = model_config.num_layers
        self.device = device
        
        # Determine dimension mode
        if hasattr(model_config, 'coor_input_dim'):
            self.dim_mode = model_config.coor_input_dim
        else:
            # Fallback or error? Assuming defaults
            # Try to infer from model name if possible, or default to 1
            if '1d' in model_config.model_name.lower():
                self.dim_mode = 1
            elif '2d' in model_config.model_name.lower():
                self.dim_mode = 2
            elif '3d' in model_config.model_name.lower():
                self.dim_mode = 3
            else:
                raise ValueError("Could not determine spatial dimension. Please specify 'coor_input_dim' in model_config.")

        self.output_ops = {
            1: nn.Conv1d,
            2: nn.Conv2d,
            3: nn.Conv3d
        }
        
        if self.dim_mode not in self.output_ops:
             raise ValueError(f"Invalid dimension mode: {self.dim_mode}. Must be 1, 2, or 3.")

        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.rnn_input_dim if i == 0 else self.rnn_hidden_dim

            cell_list.append(ConvLSTMCell(
                input_dim=cur_input_dim,
                hidden_dim=self.rnn_hidden_dim,
                kernel_size=self.rnn_kernel_size,
                padding=self.rnn_kernel_size//2,
                bias=self.bias,
                device=self.device,
                dim_mode=self.dim_mode))
        self.cell_list = nn.ModuleList(cell_list)

        self.output_layer = self.output_ops[self.dim_mode](
            self.rnn_hidden_dim, 
            self.rnn_output_dim, 
            kernel_size=self.rnn_kernel_size, 
            stride=1, 
            padding=self.rnn_kernel_size//2, 
            padding_mode='circular').to(device)

    def model_forward(self, x, static_data):
        input_tensor = static_data[0]
        init_hidden_state = x

        output_inner = []
        internal_state = []
        
        # Determine hidden shape based on init_hidden_state if available, or x
        if init_hidden_state is None:
            # This case might need careful handling of dimensions
            # But usually init_hidden_state is passed
            hidden_shape = list(x.shape)
            hidden_shape[1] = self.rnn_hidden_dim
        else:
            hidden_shape = list(init_hidden_state.shape)
            # Ensure hidden_dim is correct in shape for init
            hidden_shape[1] = self.rnn_hidden_dim

        for t in range(self.seq_len):
            cur_layer_input = input_tensor
            for layer_idx in range(self.num_layers):                    
                if t == 0:
                    h, c = self._init_hidden(hidden_shape=hidden_shape, init_hidden_state=init_hidden_state)[layer_idx]
                    internal_state.append([h, c])

                h, c = internal_state[layer_idx]
                cur_layer_input, new_c = self.cell_list[layer_idx](input_tensor=cur_layer_input,
                                                    cur_state=[h, c])
                internal_state[layer_idx] = [cur_layer_input, new_c]
                
            output = self.output_layer(h)
            output_inner.append(output)

        layer_output = torch.stack(output_inner, dim=1)

        return layer_output

    def _init_hidden(self, hidden_shape, init_hidden_state=None):
        init_states = []
        for i in range(self.num_layers):
            init_state = self.cell_list[i].init_hidden(hidden_shape)
            if init_hidden_state is not None:
                # Check dimensions
                if len(init_hidden_state.shape) != len(init_state[0].shape):
                     # Expected dims: Batch + Channel + Spatial...
                     # 1D: B, C, X (3 dims)
                     # 2D: B, C, X, Y (4 dims)
                     # 3D: B, C, X, Y, Z (5 dims)
                     expected_dims = 2 + self.dim_mode
                     raise Exception(
                        rf"Shape of 'init_hidden_state' should be {expected_dims} dimensions, instead of {init_hidden_state.shape}.")
                init_state = (init_hidden_state, init_hidden_state)
            init_states.append(init_state)
        return init_states
    
    def forward(self, x, static_data):
        # special part for ConvLSTM
        init_hidden_state = x
        if init_hidden_state.shape[1] < self.rnn_hidden_dim:
            # Need to repeat/cat to match hidden dim
            # Calculate repetition factor based on input size
            # The logic in original files: 
            # 1D: init_hidden_state.repeat(1, self.rnn_hidden_dim // ((1 + self.involve_history_step) * self.channel_dim), 1)
            # 2D: init_hidden_state.repeat(1, ..., 1, 1)
            # 3D: init_hidden_state.repeat(1, ..., 1, 1, 1)
            
            repeat_tuple = [1] * (self.dim_mode + 2) # [1, 1, 1...]
            # repeat_tuple[0] is batch, always 1
            # repeat_tuple[1] is channel
            repeat_tuple[1] = self.rnn_hidden_dim // ((1 + self.involve_history_step) * self.channel_dim)
            
            init_hidden_state = init_hidden_state.repeat(*repeat_tuple)
            
            if init_hidden_state.shape[1] != self.rnn_hidden_dim:
                init_hidden_state = torch.cat(
                    (init_hidden_state, init_hidden_state[:, -(self.rnn_hidden_dim - init_hidden_state.shape[1]):]), dim=1)
        else:
            init_hidden_state = init_hidden_state[:, -self.rnn_hidden_dim:]
        return self.model_forward(init_hidden_state, static_data)
