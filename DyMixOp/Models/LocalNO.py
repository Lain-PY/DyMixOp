import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union
from neuralop.models import LocalNO as LNO

# Assume LocalNO class is already defined in the context
# If LocalNO is defined in another file, uncomment and modify the import path
# from your_module import LocalNO 

class LocalNO(LNO):
    """
    Autoregressive wrapper for LocalNO.
    
    This class inherits from LocalNO and is designed to mimic the input/output interface 
    and autoregressive behavior of FNO2d. It is entirely driven by model_config.
    """

    def __init__(self, model_config, device):
        """
        Initialize the LocalNO2d_AR model.

        Args:
            model_config: Configuration object, must contain the following attributes:
                - global_modes (int): Number of Fourier modes (modes1, modes2)
                - hidden_dim (int): Hidden layer width
                - num_layers (int): Number of LocalNOBlock layers
                - input_dim (int): Dimension of physical variables (e.g., 3 for u, v, p)
                - coor_input_dim (int): Dimension of static/coordinate data (e.g., 2 for x, y)
                - inp_involve_history_step (int): Number of history steps included in input (0 excludes current step, 1 includes 1 history step)
                - ar_nseq_train (int): Number of autoregressive steps during training
                - ar_nseq_test (int): Number of autoregressive steps during testing
                # Note: If model_config does not have spatial_res (H, W), default_in_shape must be manually specified
            device: torch.device
        """
        
        self.config = model_config
        self.device = device

        # 1. Extract parameters
        # Dimension of static/coordinate data
        self.coor_dim = model_config.coor_input_dim
        # Number of modes (assuming H and W directions are the same)
        n_modes = [model_config.global_modes] * self.coor_dim
        if self.coor_dim < 2:
            n_modes += [1]
        
        if isinstance(model_config.global_modes, list):
            n_modes = model_config.global_modes
            if len(n_modes) != self.coor_dim:
                raise ValueError("Length of global_modes must be equal to coor_input_dim")
        elif isinstance(model_config.global_modes, int):
            n_modes = [model_config.global_modes] * self.coor_dim
            if self.coor_dim < 2:
                n_modes += [1]
        else:
            raise ValueError("global_modes must be either an int or a list of ints")
        
        # Dimension of physical variables
        self.input_num = model_config.input_dim
        # Length of history window (Total steps in input) = History steps + 1 (Current step)
        # FNO2d logic: input_num * (1 + inp_involve_history_step)
        self.input_time_steps = 1 + getattr(model_config, 'inp_involve_history_step', 0)
        
        # Total length of prediction sequence (Train + Test)
        self.seq_len = model_config.ar_nseq_train + model_config.ar_nseq_test

        # 2. Calculate actual input/output channels for LocalNO
        # Input channels = (Physical variables * Time window) + Static coordinates
        # This is crucial: since we manually concatenate coordinates in forward, it's part of the network input
        actual_in_channels = (self.input_num * self.input_time_steps) + self.coor_dim
        
        # Output channels = Number of physical variables for single-step prediction
        actual_out_channels = self.input_num
        
        hidden_channels = model_config.hidden_dim
        self.n_layers = model_config.num_layers

        # 3. Determine default input shape (LocalNO needs this parameter for DISCO convolution)
        # If no explicit spatial resolution in config, give a default value (64, 64) or user needs to ensure config contains this
        # It is recommended to add spatial_shape attribute to model_config
        default_in_shape = getattr(model_config, 'spatial_shape', [64, 64])
        disco_layers = getattr(model_config, 'disco_layers', True)
        if len(default_in_shape) < 2:
            default_in_shape += [1]
        self.default_in_shape = default_in_shape

        # 4. Initialize parent class LocalNO
        super().__init__(
            n_modes=n_modes,
            in_channels=actual_in_channels,
            out_channels=actual_out_channels,
            hidden_channels=hidden_channels,
            n_layers=self.n_layers,
            default_in_shape=default_in_shape,
            # Set positional_embedding to None, because we will manually pass static_data (coordinates) in forward
            # To maintain consistency with FNO2d behavior
            positional_embedding=None, 
            # Other parameters can be read from config or kept as default
            lifting_channel_ratio=2,
            projection_channel_ratio=2,
            disco_layers=disco_layers # when encountering the 1D case, the disco layer equals to the difference layer and then we set it to False.
        )
        
        # Move model to specified device
        self.to(device)

    def forward(self, x, static_data):
        """
        Autoregressive forward pass.

        Args:
            x (torch.Tensor): Temporal snapshots input.
                Shape: (Batch, Time_Steps * Channels, Height, Width)
                Note: Channels here is input_dim. If multiple time steps are stacked, they are usually flattened in dim 1.
            static_data (List[torch.Tensor]): List of static data.
                static_data[0] Shape: (Batch, Coor_Channels, Height, Width)

        Returns:
            torch.Tensor: Predicted future time step sequence.
                Shape: (Batch, Seq_Len, Channels, Height, Width)
        """
        
        # Prepare static data (coordinates)
        coor_input = static_data[0] # (B, C_coor, H, W)
        
        # Initialize current input window
        x_in = x 
        # 1. Check if spatial dim is 1 and then extend it to 2D
        if x.dim() == 3:
            coor_input = coor_input.unsqueeze(-1)
            x_in = x_in.unsqueeze(-1)
        
        # List to store prediction results for each step
        output_list = []

        # Autoregressive loop
        for k in range(self.seq_len):
            # 1. Concatenate input: [history physical variables, static coordinates]
            # Input tensor shape for LocalNO: (B, (T*C_p + C_s), H, W)
            model_input = torch.cat((x_in, coor_input), dim=1)

            # 2. Call parent forward pass (LocalNO core calculation)
            # Output shape: (B, C_p, H, W) -> Single step prediction
            x_pred = super().forward(model_input, output_shape=[tuple(self.default_in_shape)]*self.n_layers)
            
            # 3. Collect results
            output_list.append(x_pred)

            # 4. Update sliding window (Sliding Window). Logic referenced from FNO2d.
            # If input contains multiple time steps, remove the oldest step and append the newest prediction
            # x_in shape assumption: (B, t0_var1, t0_var2, t1_var1, t1_var2..., H, W) flattened in dim 1
            if self.input_time_steps > 1:
                # Remove the oldest frame (i.e., remove the first input_num channels)
                # Concatenate the latest prediction
                x_in = torch.cat((x_in[:, self.input_num:], x_pred), dim=1)
            else:
                # If only the previous step is needed as input
                x_in = x_pred

        # 5. Stack outputs
        # Result shape: (Batch, Seq_Len, Channels, Height, Width)
        out = torch.stack(output_list, dim=1)

        if x.dim() == 3:
            # Squeeze spatial dims
            out = out.squeeze(-1)
        
        return out

    # If compatibility with old interface or original forward functionality is needed, use alias
    # def model_forward(self, x):
    #     return super().forward(x)