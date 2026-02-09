import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union
from neuralop.models import UNO as UNO_BASE

# Assume UNO class is already defined in the context
# If UNO is defined in another file, uncomment and modify the import path
# from your_module import UNO 

def generate_uno_config(n_layers: int, base_channels: int, base_modes: int, spatial_dim: int, uno_out_channels=None, uno_scalings=None):
    """
    Generate automatic configuration for U-shaped Neural Operator (UNO) model layers.
    
    Implements an aggressive scaling strategy for multi-scale neural operators:
    - Downsampling phase: Reduce resolution by 0.5x, increase channels by 2x, decrease modes by 0.5x
    - Bottleneck layer: Maintain current configuration at the deepest layer
    - Upsampling phase: Increase resolution by 2.0x, decrease channels by 0.5x, increase modes by 2.0x
    
    This approach efficiently balances computational complexity with model capacity
    by progressively changing resolution, channels, and spectral modes across layers.
    
    Args:
        n_layers (int): 
            Total number of layers in the network. 
            For optimal U-Net structure, odd numbers (3, 5, 7, ...) are recommended.
        base_channels (int): 
            Initial hidden layer width (hidden_dim) that serves as the base for channel scaling.
        base_modes (int): 
            Number of Fourier modes at the base resolution (global_modes) for spectral operations.
        spatial_dim (int): 
            Spatial dimensionality of the data (1 for 1D, 2 for 2D, 3 for 3D).
        uno_out_channels (list, optional): 
            Custom output channels per layer. Auto-generated if not provided.
        uno_scalings (list, optional): 
            Custom scaling factors per layer. Auto-generated if not provided.

    Returns:
        dict: Configuration dictionary containing:
            - 'uno_scalings': List of scaling factors for each layer
            - 'uno_out_channels': List of output channels for each layer  
            - 'uno_n_modes': List of Fourier modes for each layer
    """
    
    # 1. Determine network structure (Down -> Mid -> Up)
    # For even layer counts, equally distribute to downsampling and upsampling with no middle layer
    n_down = n_layers // 2
    n_up = n_layers // 2
    n_mid = n_layers % 2  # Odd count includes one bottleneck layer in the middle

    scalings = []
    out_channels = []
    n_modes = []

    # Track current state variables during network construction
    current_res_scale = 1.0  # Resolution scale factor relative to input
    current_ch_scale = 1     # Channel multiplier relative to base_channels

    # --- Phase 1: Encoding (Downsampling) ---
    for i in range(n_down):
        # Downsample by factor of 0.5
        if i % 2 != 0:
            scalings.append([0.5] * spatial_dim)
        else:
            scalings.append([1.0] * spatial_dim)
        
        # Apply aggressive strategy: halve resolution, double channels, halve modes
        if i % 2 != 0:
            current_res_scale *= 0.5
        current_ch_scale *= 2
        
        out_channels.append(base_channels * current_ch_scale)
        
        # Compute modes for current resolution (rounded down, minimum 1)
        # Note: Mode count is constrained by Nyquist frequency (half the resolution)
        # to ensure we don't exceed the maximum resolvable frequency in spectral space.
        if isinstance(base_modes, int):
            layer_mode = max(1, int(base_modes * current_res_scale))
            n_modes.append([layer_mode] * spatial_dim)

    # --- Phase 2: Bottleneck (Middle) ---
    if n_mid > 0:
        # Maintain current scale (1.0x)
        scalings.append([1.0] * spatial_dim)
        
        # Preserve the deepest layer configuration
        out_channels.append(base_channels * current_ch_scale)
        
        if isinstance(base_modes, int):
            layer_mode = max(1, int(base_modes * current_res_scale))
            n_modes.append([layer_mode] * spatial_dim)

    # --- Phase 3: Decoding (Upsampling) ---
    up_scaling = []
    for i in range(n_up):
        # Upsample by factor of 2.0
        if i % 2 != 0:
            up_scaling.append([2.0] * spatial_dim)
        else:
            up_scaling.append([1.0] * spatial_dim)
        
        # Apply reverse aggressive strategy: double resolution, halve channels, double modes
        if i % 2 != 0:
            current_res_scale *= 2.0
        current_ch_scale //= 2  # Use integer division for channel count
        
        out_channels.append(base_channels * current_ch_scale)
        
        if isinstance(base_modes, int):
            layer_mode = max(1, int(base_modes * current_res_scale))
            n_modes.append([layer_mode] * spatial_dim)
    scalings.extend(up_scaling[::-1])
    
    # Override with user-provided configurations if available; otherwise, use auto-generated defaults
    if isinstance(base_modes, list):
        if len(base_modes) == n_layers:
            n_modes = base_modes
        else:
            raise ValueError("base_modes must be an int or a list of ints with length equal to n_layers")

    if uno_out_channels is not None:
        if len(uno_out_channels) == n_layers:
            out_channels = uno_out_channels
        else:
            raise ValueError("uno_out_channels must be a list of ints with length equal to n_layers")
        
    if uno_scalings is not None:
        if len(uno_scalings) == n_layers:
            scalings = uno_scalings
        else:
            raise ValueError("uno_scalings must be a list of lists with length equal to n_layers")

    return {
        "uno_scalings": scalings,
        "uno_out_channels": out_channels,
        "uno_n_modes": n_modes
    }

class UNO(UNO_BASE):
    """
    Autoregressive wrapper for UNO (U-Shaped Neural Operator).
    
    This class inherits from UNO and adheres to the input/output interface 
    required by the configuration-driven training pipeline.
    """

    def __init__(self, model_config, device):
        """
        Initialize the UNO model.

        Args:
            model_config: Configuration object.
                Required basic attributes:
                - global_modes (int): Base number of Fourier modes.
                - hidden_dim (int): Base hidden layer width.
                - num_layers (int): Number of Fourier Layers.
                - input_dim (int): Dimension of physical variables.
                - coor_input_dim (int): Dimension of static/coordinate data.
                - inp_involve_history_step (int): History steps.
                - ar_nseq_train (int): AR steps for training.
                - ar_nseq_test (int): AR steps for testing.
                
                Optional/New attributes:
                - spatial_dim (int): Spatial dimension of the data (1, 2, or 3). Default is 2.
                
                Optional UNO-specific attributes (will be auto-generated if missing):
                - uno_out_channels (list): Output channels for each layer.
                - global_modes (list): Modes for each layer.
                - uno_scalings (list): Scaling factors for each layer.
                - horizontal_skips_map (dict): Skip connection map.
            device: torch.device
        """
        
        self.config = model_config
        self.device = device

        # 1. Extract Basic Parameters
        n_layers = model_config.num_layers
        base_modes = model_config.global_modes
        base_width = model_config.hidden_dim
        
        self.input_num = model_config.input_dim
        self.coor_dim = model_config.coor_input_dim
        self.input_time_steps = 1 + getattr(model_config, 'inp_involve_history_step', 0)
        self.seq_len = model_config.ar_nseq_train + model_config.ar_nseq_test

        # 2. Calculate Actual Input/Output Channels
        # Input = (Physical Vars * Time Steps) + Coordinates
        actual_in_channels = (self.input_num * self.input_time_steps) + self.coor_dim
        actual_out_channels = self.input_num # Single step prediction

        # 3. Handle UNO Specific List Parameters
        # Use user-provided configurations if available; otherwise, use auto-generated defaults.
        self.uno_out_channels = getattr(model_config, 'uno_out_channels', None)
        self.uno_scalings = getattr(model_config, 'uno_scalings', None)
        
        # Generate UNO configuration using the new function
        uno_config = generate_uno_config(n_layers, base_width, base_modes, self.coor_dim, self.uno_out_channels, self.uno_scalings)
        
        # modes: [[m, ...], [m, ...], ...] depending on coor_dim
        global_modes = uno_config["uno_n_modes"]
        # channels: [w, w, ...]
        uno_out_channels = uno_config["uno_out_channels"]
        # scalings: [[1,...], [1,...], ...] depending on coor_dim
        uno_scalings = uno_config["uno_scalings"]

        # skips map
        horizontal_skips_map = getattr(model_config, 'horizontal_skips_map', None)

        # 4. Initialize Parent Class (UNO)
        super().__init__(
            in_channels=actual_in_channels,
            out_channels=actual_out_channels,
            hidden_channels=base_width,
            n_layers=n_layers,
            uno_out_channels=uno_out_channels,
            uno_n_modes=global_modes,
            uno_scalings=uno_scalings,
            horizontal_skips_map=horizontal_skips_map,
            # Set positional_embedding to None because coordinates are passed via static_data
            positional_embedding=None,
            # Pass other optional params if they exist in config, else defaults
            lifting_channels=2*base_width,
            projection_channels=2*base_width,
            channel_mlp_skip="linear",
        )
        
        self.to(device)

    def forward(self, x, static_data):
        """
        Autoregressive forward pass.

        Args:
            x (torch.Tensor): Temporal snapshots input.
                Shape: (Batch, Time_Steps * Channels, D1, [D2, ...])
            static_data (List[torch.Tensor]): List of static data.
                static_data[0] Shape: (Batch, Coor_Channels, D1, [D2, ...])

        Returns:
            torch.Tensor: Predicted future time step sequence.
                Shape: (Batch, Seq_Len, Channels, D1, [D2, ...])
        """
        
        # Prepare static data (coordinates)
        coor_input = static_data[0] # (B, C_coor, ...)
        
        # Initialize current input window
        x_in = x 
        
        output_list = []

        # Autoregressive loop
        for k in range(self.seq_len):
            # 1. Concatenate input: [history physical variables, static coordinates]
            model_input = torch.cat((x_in, coor_input), dim=1)

            # 2. Call parent forward pass (UNO core)
            # Output shape: (B, C_p, ...)
            x_pred = super().forward(model_input)
            
            # 3. Collect results
            output_list.append(x_pred)

            # 4. Update sliding window
            if self.input_time_steps > 1:
                # Shift window: remove oldest timestep and append new prediction
                x_in = torch.cat((x_in[:, self.input_num:], x_pred), dim=1)
            else:
                x_in = x_pred

        # 5. Stack outputs
        out = torch.stack(output_list, dim=1)
        
        return out

    # Alias for compatibility
    def model_forward(self, x):
        return super().forward(x)