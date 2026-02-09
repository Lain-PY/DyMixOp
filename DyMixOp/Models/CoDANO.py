import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union
from neuralop.models import CODANO as CODANO_BASE

# Assume CODANO class is already defined in the context
# If CODANO is defined in another file, uncomment and modify the import path
# from your_module import CODANO

class CoDANO(CODANO_BASE):
    """
    Autoregressive wrapper for CoDANO (Codomain Attention Neural Operators).
    
    This class inherits from CODANO and adapts it to the configuration-driven 
    autoregressive training pipeline.
    """

    def __init__(self, model_config, device):
        """
        Initialize the CoDANO_AR model.

        Args:
            model_config: Configuration object.
                Required basic attributes:
                - global_modes (int): Base number of Fourier modes.
                - hidden_dim (int): Hidden variable codimension (width of tokens).
                - num_layers (int): Number of CoDA Layers.
                - input_dim (int): Dimension of physical variables (e.g., 3 for u, v, p).
                - coor_input_dim (int): Dimension of static/coordinate data.
                - inp_involve_history_step (int): History steps.
                - ar_nseq_train (int): AR steps for training.
                - ar_nseq_test (int): AR steps for testing.
                
                Optional attributes:
                - n_heads (int or List[int]): Number of attention heads.
            device: torch.device
        """
        
        self.config = model_config
        self.device = device

        # 1. Extract Basic Parameters
        n_layers = model_config.num_layers
        base_modes = model_config.global_modes
        hidden_dim = model_config.hidden_dim
        
        self.input_num = model_config.input_dim
        self.coor_dim = model_config.coor_input_dim
        self.input_time_steps = 1 + getattr(model_config, 'inp_involve_history_step', 0)
        self.seq_len = model_config.ar_nseq_train + model_config.ar_nseq_test
        self.out_dim = model_config.output_dim * self.seq_len

        # 2. Construct CoDANO specific parameters
        
        # Modes: [[m, m], ...] for 2D, or adapted for n_dim
        if hasattr(model_config, 'codano_n_modes') and model_config.codano_n_modes is not None:
            n_modes = model_config.codano_n_modes
        else:
            if self.coor_dim == 1:
                n_modes = [[base_modes, 1]] * n_layers
            else:
                n_modes = [[base_modes] * self.coor_dim] * n_layers

        # Heads: Default to 1 if not specified
        if hasattr(model_config, 'n_heads'):
             n_heads = model_config.n_heads
             if isinstance(n_heads, int):
                 n_heads = [n_heads] * n_layers
        else:
            n_heads = [1] * n_layers

        # 3. Initialize Parent Class (CODANO)
        # Note: We set static_channel_dim=0 and use_positional_encoding=False
        # because we will manually concatenate coordinates to the input 'x' in forward().
        # This treats coordinates as just another "variable" token in the attention mechanism.
        super().__init__(
            n_layers=n_layers,
            n_modes=n_modes,
            hidden_variable_codimension=hidden_dim, # hidden_dim // (self.input_time_steps * self.input_num + self.coor_dim), # Map hidden_dim to hidden_variable_codimension
            output_variable_codimension=self.seq_len,
            n_heads=n_heads,
            lifting_channels=2*hidden_dim,
            projection_channels=2*hidden_dim,
            per_layer_scaling_factors=[[1] * len(n_modes[0])] * n_layers,
            attention_scaling_factors=[1] * n_layers,

        )
        
        self.to(device)

    def forward(self, x, static_data):
        """
        Autoregressive forward pass.

        Args:
            x (torch.Tensor): Temporal snapshots input.
                Shape: (Batch, Time_Steps * Channels, D1, [D2, ...])
                CoDANO treats 'Time_Steps * Channels' as the number of input variables.
            static_data (List[torch.Tensor]): List of static data.
                static_data[0] Shape: (Batch, Coor_Channels, D1, [D2, ...])

        Returns:
            torch.Tensor: Predicted future time step sequence.
                Shape: (Batch, Seq_Len, Channels, D1, [D2, ...])
        """
        
        # Prepare static data (coordinates)
        coor_input = static_data[0] # (B, C_coor, ...)
        
        # Initialize current input window
        x_in = x[:, -self.input_num:]
        if x.dim() == 3:
            coor_input = coor_input.unsqueeze(-1)
            x_in = x_in.unsqueeze(-1)
        
        output_list = []

        # non-autoregressive prediction loop
        for k in range(1): # use self.seq_len if autoregressive, 1 if non-autoregressive
            # 1. Concatenate input: [history variables, static coordinates]
            # This combined tensor creates the set of "variables" for CoDANO attention.
            model_input = x_in # torch.cat((x_in, coor_input), dim=1)

            # 2. Call parent forward pass
            # CoDANO returns a tensor with the same number of variables as input (if out_codim=1).
            # Shape: (B, (T*C + C_coor), ...)
            x_pred = super().forward(model_input)
            
            # 4. Collect results
            output_list.append(x_pred)

            # 5. Update sliding window
            if self.input_time_steps > 1:
                # Remove oldest time step, append new prediction
                x_in = torch.cat((x_in[:, self.input_num:], x_pred), dim=1)
            else:
                x_in = x_pred

        # 6. Stack outputs
        out = torch.stack(output_list, dim=1).reshape(x_in.shape[0], self.seq_len, -1, *x_in.shape[2:])

        if x.dim() == 3:
            out = out.squeeze(-1)
        
        return out

    # Alias for compatibility
    def model_forward(self, x):
        # Note: This simple alias might fail if x doesn't contain coordinates.
        # It's better to rely on the main forward.
        return super().forward(x)