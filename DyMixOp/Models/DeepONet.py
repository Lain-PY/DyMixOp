import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

torch.manual_seed(0)
np.random.seed(0)


class DeepONet(nn.Module):
    """Deep operator network in
    the format of Cartesian product.
    Args:
        pod_basis: POD basis used in the trunk net.
        layer_sizes_branch: A list of integers as the width of a fully connected network,
            or `(dim, f)` where `dim` is the input dimension and `f` is a network
            function. The width of the last layer in the branch and trunk net should be
            equal.
        activation: If `activation` is a ``string``, then the same activation is used in
            both trunk and branch nets. If `activation` is a ``dict``, then the trunk
            net uses the activation `activation["trunk"]`, and the branch net uses
            `activation["branch"]`.
        layer_sizes_trunk (list): A list of integers as the width of a fully connected
            network. If ``None``, then only use POD basis as the trunk net.
    References:
        `L. Lu, X. Meng, S. Cai, Z. Mao, S. Goswami, Z. Zhang, & G. E. Karniadakis. A
        comprehensive and fair comparison of two neural operators (with practical
        extensions) based on FAIR data. arXiv preprint arXiv:2111.05512, 2021
        <https://arxiv.org/abs/2111.05512>`_.
    """

    def __init__(self, model_config, device):
        super().__init__()
        self.input_dim = model_config.input_dim
        self.hidden_dim = model_config.hidden_dim
        self.output_dim = model_config.output_dim

        self.involve_history_step = model_config.inp_involve_history_step
        self.seq_len = model_config.ar_nseq_train + model_config.ar_nseq_test
        
        # Determine dimension mode
        if hasattr(model_config, 'coor_input_dim'):
            self.dim_mode = model_config.coor_input_dim
        else:
            # Fallback
            if '1d' in model_config.model_name.lower():
                self.dim_mode = 1
            elif '2d' in model_config.model_name.lower():
                self.dim_mode = 2
            elif '3d' in model_config.model_name.lower():
                self.dim_mode = 3
            else:
                 # Default to 1 or raise error?
                 self.dim_mode = 1

        self.len_x = int(np.ceil(model_config.res_x / model_config.res_step))
        self.len_y = int(np.ceil(model_config.res_y / model_config.res_step))
        self.len_z = int(np.ceil(model_config.res_z / model_config.res_step))
        
        # Used for 2D/3D but harmless for 1D
        self.len_xyz = self.len_x * self.len_y * self.len_z

        self.num_layers = model_config.num_layers

        min_res_x = int(np.ceil(self.len_x / (2 * self.num_layers)))
        min_res_y = int(np.ceil(self.len_y / (2 * self.num_layers)))
        min_res_z = int(np.ceil(self.len_z / (2 * self.num_layers)))
        
        min_res = [min_res_x, min_res_y, min_res_z]
        self.res_feat = model_config.res_feature_dim
        
        # Adjust res_feat based on dimensions
        # The original code iterates len(self.res_feat) and uses min_res[i]
        # In 1D config, res_feat usually has 1 element? Or multiple?
        # DeepONet1d uses self.res_feat[0]
        # DeepONet2d uses self.res_feat[0], [1]
        # DeepONet3d uses self.res_feat[0], [1], [2]
        # We need to ensure res_feat is handled correctly up to dim_mode
        
        for i in range(len(self.res_feat)):
            if i < 3: # prevent index error if res_feat is longer for some reason
                self.res_feat[i] = min(self.res_feat[i], min_res[i])

        self.kernel_size = model_config.kernel_size
        self.stride = model_config.stride
        
        # Setup operations based on dimension
        if self.dim_mode == 1:
            self.conv_op = nn.Conv1d
            self.batchnorm_op = nn.BatchNorm1d
            self.adaptive_pool_op = nn.AdaptiveAvgPool1d
            self.flatten_dim = self.res_feat[0]
        elif self.dim_mode == 2:
            self.conv_op = nn.Conv2d
            self.batchnorm_op = nn.BatchNorm2d
            self.adaptive_pool_op = nn.AdaptiveAvgPool2d
            self.flatten_dim = self.res_feat[0] * self.res_feat[1]
        elif self.dim_mode == 3:
            self.conv_op = nn.Conv3d
            self.batchnorm_op = nn.BatchNorm3d
            self.adaptive_pool_op = nn.AdaptiveAvgPool3d
            self.flatten_dim = self.res_feat[0] * self.res_feat[1] * self.res_feat[2]
        else:
             raise ValueError(f"Invalid dimension mode: {self.dim_mode}")


        # Create multiple branches, one for each output dimension
        self.branches = nn.ModuleList()
        for _ in range(self.output_dim):
            branch_list = []
            for i in range(self.num_layers):
                temp_pre_hidden_dim = self.hidden_dim * (i) // self.num_layers
                temp_hidden_dim = self.hidden_dim * (i+1) // self.num_layers
                if i == 0:
                    branch_list.append(
                        self.conv_op(self.input_dim*(self.involve_history_step+1), temp_hidden_dim, 
                                      kernel_size=self.kernel_size, stride=self.stride, padding=self.kernel_size // 2,
                                      padding_mode='circular').to(device))
                else:
                    branch_list.append(self.conv_op(temp_pre_hidden_dim, temp_hidden_dim, 
                                                     kernel_size=self.kernel_size, stride=self.stride, padding=self.kernel_size // 2,
                                                     padding_mode='circular').to(device))
                branch_list.append(self.batchnorm_op(temp_hidden_dim).to(device))
                branch_list.append(torch.nn.GELU().to(device))

            branch = torch.nn.Sequential(
                *branch_list,
                self.adaptive_pool_op(self.res_feat[:self.dim_mode] if len(self.res_feat) >= self.dim_mode else self.res_feat), # Pass correct slice of res_feat
                torch.nn.Flatten(),
                torch.nn.Linear(self.flatten_dim * self.hidden_dim, self.flatten_dim // 2 * self.hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(self.flatten_dim // 2 * self.hidden_dim, 1 * self.hidden_dim),
            ).to(device)
            
            self.branches.append(branch)

        trunk_list = []
        for i in range(self.num_layers):
            if i == 0:
                trunk_list.append(
                    torch.nn.Linear(self.dim_mode, self.hidden_dim//(2**(self.num_layers-1))).to(device))  # Input is dim_mode (1, 2, or 3)
            else:
                trunk_list.append(
                    torch.nn.Linear(self.hidden_dim*2**(i-1)//(2**(self.num_layers-1)), self.hidden_dim*2**i//(2**(self.num_layers-1))).to(device))
            trunk_list.append(torch.nn.GELU().to(device))

        self.trunk = torch.nn.Sequential(
            *trunk_list,
        ).to(device)

        self.b = torch.nn.ParameterList([
            torch.nn.Parameter(torch.tensor(0.0, device=device))
            for _ in range(self.output_dim)
        ])
        
        self.device = device


    def model_forward(self, inputs, x_cor):
        x_list = []
        x_in = inputs
        # Reshape x_cor based on dim_mode
        # x_cor comes in as various shapes, we need (Dimension, NumPoints) then permute to (NumPoints, Dimension)
        # Original:
        # 1D: x_cor.reshape(1, -1).permute(1, 0)
        # 2D: x_cor.reshape(2, -1).permute(1, 0)
        # 3D: x_cor.reshape(3, -1).permute(1, 0)
        
        x_cor = x_cor.reshape(self.dim_mode, -1).permute(1, 0)

        output_shape = list(inputs.shape)
        output_shape[1] = self.output_dim 
        for i in range(self.seq_len):
            x_func = x_in
            x_loc = x_cor

            # Process each branch and combine results
            outputs = []
            for j in range(self.output_dim):
                x_func_j = self.branches[j](x_func)
                x_loc_j = self.trunk(x_loc)
                x_j = torch.einsum("bi,ni->bn", x_func_j, x_loc_j) / (self.hidden_dim ** 2)
                x_j += self.b[j]
                outputs.append(x_j)

            # Concatenate all outputs
            x = torch.cat(outputs, dim=1)
            x = x.reshape(output_shape)

            x_list.append(x)

            x_in = torch.cat((x_in[:, self.input_dim:], x), dim=1)

        x = torch.stack(x_list, dim=1)

        return x

    def forward(self, input_data, static_data):
        # special part for deeponet
        X = input_data
        
        # Handle static_data extraction
        # Original:
        # 1D: X_cor = static_data[0][0, 0]
        # 2D: X_cor = static_data[0][0] 
        # 3D: X_cor = static_data[0][0]
        
        # Wait, for 1D it was static_data[0][0, 0] ?? 
        # DeepONet1d.py: X_cor = static_data[0][0, 0]
        # DeepONet2d.py: X_cor = static_data[0][0]
        # DeepONet3d.py: X_cor = static_data[0][0]
        
        # Let's inspect infernece/trainer logic or static data generator.
        # utils.generate_coor_input returns:
        # 1D: [batch, 1, res_x]
        # 2D: [batch, 2, res_x, res_y]
        # 3D: [batch, 3, res_x, res_y, res_z]
        
        # static_data[0] is this tensor for batch. 
        # static_data[0][0] is [1, res_x] or [2, X, Y] or [3, X, Y, Z] (First item in batch)
        
        # In 1D: static_data[0][0, 0] would be [res_x] ? 
        # Because static_data[0] is [batch, 1, res_x]. 
        # static_data[0][0] is [1, res_x].
        # static_data[0][0, 0] is [res_x]. 
        # And model_forward reshapes to (1, -1) -> (1, res_x) -> permute -> (res_x, 1). Trunk expects (N, 1). Correct.
        
        # In 2D: static_data[0] is [batch, 2, X, Y].
        # static_data[0][0] is [2, X, Y].
        # model_forward checks x_cor.reshape(2, -1) -> (2, X*Y) -> permutes -> (X*Y, 2). Trunk expects (N, 2). Correct.
        
        # In 3D: static_data[0] is [batch, 3, X, Y, Z].
        # static_data[0][0] is [3, X, Y, Z].
        # model_forward checks x_cor.reshape(3, -1) -> (3, X*Y*Z) -> permutes -> (X*Y*Z, 3). Trunk expects (N, 3). Correct.
        
        # So for 1D, we access [0, 0] because we want to remove the 'channel' dim of coordinate which is 1.
        # But for 2D/3D we pass [0] which has channel dim 2 or 3.
        
        # Actually... 
        # If I pass static_data[0][0] in 1D, it is [1, res_x]. 
        # Reshape(1, -1) -> (1, res_x). Perfect.
        # So why did 1D code use [0, 0]? 
        # Maybe [0, 0] extracts inner tensor.
        # If I use static_data[0][0] for 1D, it works too with reshape(1, -1).
        
        # Let's verify:
        # 1D tensor T shape [1, 100]. T.reshape(1, -1) -> [1, 100].
        # 1D tensor T shape [100]. T.reshape(1, -1) -> [1, 100].
        # Both work.
        
        # However, to be safe and consistent with previous code's intent (getting the coordinate grid):
        # I'll just use static_data[0][0] and rely on reshape.
        
        X_cor = static_data[0][0]
        
        return self.model_forward(X, X_cor)
