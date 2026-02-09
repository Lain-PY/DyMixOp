import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union
from neuralop.models import FNO as FNO_BASE

# Assume FNO class is already defined in the context
# If FNO is defined in another file, uncomment and modify the import path
# from your_module import FNO 

class FNO(FNO_BASE):
    """
    Autoregressive wrapper for FNO.
    
    This class inherits from FNO and is designed to mimic the input/output interface 
    and autoregressive behavior required by the configuration-driven training pipeline.
    """

    def __init__(self, model_config, device):
        """
        Initialize the FNO_AR model.

        Args:
            model_config: Configuration object, must contain the following attributes:
                - global_modes (int): Number of Fourier modes (assumed equal for both dimensions in 2D)
                - hidden_dim (int): Hidden layer width
                - num_layers (int): Number of Fourier Layers
                - input_dim (int): Dimension of physical variables (e.g., 3 for u, v, p)
                - coor_input_dim (int): Dimension of static/coordinate data (e.g., 2 for x, y)
                - inp_involve_history_step (int): Number of history steps included in input
                - ar_nseq_train (int): Number of autoregressive steps during training
                - ar_nseq_test (int): Number of autoregressive steps during testing
            device: torch.device
        """
        
        self.config = model_config
        self.device = device

        # 1. Extract parameters
        # Dimension of static/coordinate data
        self.coor_dim = model_config.coor_input_dim
        # Number of modes (assuming H and W directions are the same)
        if isinstance(model_config.global_modes, list):
            n_modes = model_config.global_modes
        elif isinstance(model_config.global_modes, int):
            n_modes = [model_config.global_modes] * self.coor_dim
        else:
            raise ValueError("global_modes must be either an int or a list of ints")
        
        # Dimension of physical variables
        self.input_num = model_config.input_dim
        # Dimension of static/coordinate data
        self.coor_dim = model_config.coor_input_dim
        
        # Length of history window (Total steps in input) = History steps + 1 (Current step)
        self.input_time_steps = 1 + getattr(model_config, 'inp_involve_history_step', 0)
        
        # Total length of prediction sequence (Train + Test)
        self.seq_len = model_config.ar_nseq_train + model_config.ar_nseq_test

        # 2. Calculate actual input/output channels for FNO
        # Input channels = (Physical variables * Time window) + Static coordinates
        # The coordinates are manually concatenated in the forward pass, so they count towards in_channels
        actual_in_channels = (self.input_num * self.input_time_steps) + self.coor_dim
        
        # Output channels = Number of physical variables for single-step prediction
        actual_out_channels = self.input_num
        
        hidden_channels = model_config.hidden_dim
        self.n_layers = model_config.num_layers

        # 3. Determine default input shape
        # If no explicit spatial resolution in config, give a default value (64, 64) or user needs to ensure config contains this
        # It is recommended to add spatial_shape attribute to model_config
        default_in_shape = getattr(model_config, 'spatial_shape', [64, 64])
        self.default_in_shape = default_in_shape

        # 4. Initialize parent class FNO
        super().__init__(
            n_modes=n_modes,
            in_channels=actual_in_channels,
            out_channels=actual_out_channels,
            hidden_channels=hidden_channels,
            n_layers=self.n_layers,
            # Set positional_embedding to None because static_data (coordinates) 
            # is manually passed and concatenated in the forward method.
            positional_embedding=None, 
            # Optional: You can map other config parameters here if needed
            lifting_channel_ratio=2,
            projection_channel_ratio=2
        )
        
        # Move model to specified device
        self.to(device)

    def forward(self, x, static_data):
        """
        Autoregressive forward pass.

        Args:
            x (torch.Tensor): Temporal snapshots input.
                Shape: (Batch, Time_Steps * Channels, Height, Width)
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
        
        # List to store prediction results for each step
        output_list = []

        # Autoregressive loop
        for k in range(self.seq_len):
            # 1. Concatenate input: [history physical variables, static coordinates]
            # Input tensor shape for FNO: (B, (T*C_p + C_s), H, W)
            model_input = torch.cat((x_in, coor_input), dim=1)

            # 2. Call parent forward pass (FNO core calculation)
            # Output shape: (B, C_p, H, W) -> Single step prediction
            x_pred = super().forward(model_input, output_shape=[tuple(self.default_in_shape)]*self.n_layers)
            
            # 3. Collect results
            output_list.append(x_pred)

            # 4. Update sliding window
            # If input contains multiple time steps, remove the oldest step and append the newest prediction
            if self.input_time_steps > 1:
                # Remove the oldest frame (i.e., remove the first input_num channels)
                # Concatenate the latest prediction at the end
                x_in = torch.cat((x_in[:, self.input_num:], x_pred), dim=1)
            else:
                # If only the previous step is needed as input
                x_in = x_pred

        # 5. Stack outputs
        # Result shape: (Batch, Seq_Len, Channels, Height, Width)
        out = torch.stack(output_list, dim=1)
        
        return out

    # Alias for compatibility if needed
    def model_forward(self, x):
        return super().forward(x)