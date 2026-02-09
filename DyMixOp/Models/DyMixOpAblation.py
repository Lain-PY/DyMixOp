"""
DyMixOp Ablation Model

Configurable neural operator for ablation experiments:
1. Interaction Mechanism: mixing/adding/local_only/global_only/mixing_adding
2. Layer Composition: conv_linear, diff_linear, nonlinear branches
3. Architecture: hybrid/only_sequential/only_parallel, adaptive_gating
4. Dimension Reduction: full_reduction/no_projection/cpx_reduc_dim_vvcc
5. Residual Operator A: none/dim_cvc/dim_cc/dim_ccc
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ==============================================================================
# Helper Functions
# ==============================================================================

def get_conv_class(coor_dim: int):
    """Get appropriate Conv class for coordinate dimension."""
    return getattr(nn, f"Conv{coor_dim}d")


def make_conv(conv_class, in_ch: int, out_ch: int, ks: int, device):
    """Create circular-padded conv layer."""
    return conv_class(in_ch, out_ch, ks, 1, padding=ks//2, padding_mode='circular', device=device)


def make_conv_stack(conv_class, dims: List[int], ks: int, device):
    """Create sequential conv layers."""
    layers = [make_conv(conv_class, dims[i], dims[i+1], ks, device) for i in range(len(dims)-1)]
    return nn.Sequential(*layers) if len(layers) > 1 else layers[0]


# ==============================================================================
# Global Transformation (Fourier)
# ==============================================================================

class GlobalTransformation(nn.Module):
    """Fourier spectral convolution layer."""
    
    def __init__(self, in_ch: int, out_ch: int, modes: int, coor_dim: int, device):
        super().__init__()
        self.in_ch, self.out_ch, self.coor_dim = in_ch, out_ch, coor_dim
        scale = 1 / (in_ch * out_ch)
        
        if coor_dim == 1:
            self.modes = modes
            self.w = nn.Parameter(scale * torch.rand(in_ch, out_ch, modes, dtype=torch.cfloat, device=device))
        elif coor_dim == 2:
            self.modes = modes
            self.w1 = nn.Parameter(scale * torch.rand(in_ch, out_ch, modes, modes, dtype=torch.cfloat, device=device))
            self.w2 = nn.Parameter(scale * torch.rand(in_ch, out_ch, modes, modes, dtype=torch.cfloat, device=device))
        else:  # 3D
            self.modes = modes
            weights = [nn.Parameter(scale * torch.rand(in_ch, out_ch, modes, modes, modes, dtype=torch.cfloat, device=device)) for _ in range(4)]
            self.w1, self.w2, self.w3, self.w4 = weights
            for i, w in enumerate(weights):
                self.register_parameter(f'w{i+1}', w)
    
    def forward(self, c: torch.Tensor) -> torch.Tensor:
        c = c.contiguous()
        m = self.modes
        
        if self.coor_dim == 1:
            c_ft = torch.fft.rfft(c)
            out_ft = torch.einsum("bix,iox->box", c_ft[:, :, :m], self.w)
            return torch.fft.irfft(out_ft, n=c.size(-1))
        
        elif self.coor_dim == 2:
            c_ft = torch.fft.rfft2(c)
            out_ft = torch.zeros(c.size(0), self.out_ch, c.size(-2), c.size(-1)//2+1, dtype=torch.cfloat, device=c.device)
            out_ft[:, :, :m, :m] = torch.einsum("bixy,ioxy->boxy", c_ft[:, :, :m, :m], self.w1)
            out_ft[:, :, -m:, :m] = torch.einsum("bixy,ioxy->boxy", c_ft[:, :, -m:, :m], self.w2)
            return torch.fft.irfft2(out_ft, s=(c.size(-2), c.size(-1)))
        
        else:  # 3D
            c_ft = torch.fft.rfftn(c, dim=[-3, -2, -1])
            out_ft = torch.zeros(c.size(0), self.out_ch, c.size(-3), c.size(-2), c.size(-1)//2+1, dtype=torch.cfloat, device=c.device)
            out_ft[:, :, :m, :m, :m] = torch.einsum("bixyz,ioxyz->boxyz", c_ft[:, :, :m, :m, :m], self.w1)
            out_ft[:, :, -m:, :m, :m] = torch.einsum("bixyz,ioxyz->boxyz", c_ft[:, :, -m:, :m, :m], self.w2)
            out_ft[:, :, :m, -m:, :m] = torch.einsum("bixyz,ioxyz->boxyz", c_ft[:, :, :m, -m:, :m], self.w3)
            out_ft[:, :, -m:, -m:, :m] = torch.einsum("bixyz,ioxyz->boxyz", c_ft[:, :, -m:, -m:, :m], self.w4)
            return torch.fft.irfftn(out_ft, s=(c.size(-3), c.size(-2), c.size(-1)))


# ==============================================================================
# Local Transformation (Convolution / Finite Difference)
# ==============================================================================

class LocalTransformation(nn.Module):
    """Local transformation via convolution or finite difference."""
    
    def __init__(self, in_ch: int, out_ch: int, ks: int, coor_dim: int, device, use_diff: bool = False):
        super().__init__()
        self.use_diff = use_diff
        
        if use_diff:
            self.diff = FiniteDifferenceConv(in_ch, out_ch, coor_dim, device).to(device)
        else:
            self.conv = make_conv(get_conv_class(coor_dim), in_ch, out_ch, ks, device)
    
    def forward(self, c: torch.Tensor, grid_width: float = 1) -> torch.Tensor:
        return self.diff(c, 1) if self.use_diff else self.conv(c)


class FiniteDifferenceConv(nn.Module):
    """Finite difference convolution."""
    
    def __init__(self, in_ch: int, out_ch: int, n_dim: int, device, ks: int = 3):
        super().__init__()
        self.n_dim, self.ks, self.pad = n_dim, ks, ks // 2
        conv_class = get_conv_class(n_dim)
        self.conv_fn = getattr(F, f"conv{n_dim}d")
        self.conv = conv_class(in_ch, out_ch, ks, padding="same", padding_mode="circular", bias=False)
        self.weight = self.conv.weight
        self.device = device
    
    def forward(self, x: torch.Tensor, grid_width: float) -> torch.Tensor:
        dims = tuple(range(2, 2 + self.n_dim))
        weight_sum = self.weight.sum(dim=dims, keepdim=True)
        fused = self.weight.clone()
        
        mid = self.ks // 2
        idx = [slice(None), slice(None)] + [mid] * self.n_dim
        fused[tuple(idx)] -= weight_sum.squeeze()
        
        pad_arg = (self.pad, self.pad) * self.n_dim
        x_pad = F.pad(x, pad_arg, mode="circular")
        return self.conv_fn(x_pad, fused, bias=None, stride=1, padding=0) / grid_width


# ==============================================================================
# Local-Global Mixing Transformation (Exp 1)
# ==============================================================================

class LGMTransformation(nn.Module):
    """Local-Global mixing with configurable interaction."""
    
    def __init__(self, in_dim: int, out_dim: int, device, modes: int, ks: int, 
                 coor_dim: int, interaction: str = 'mixing', approx_nonlin: bool = True):
        super().__init__()
        self.interaction = interaction
        self.approx_nonlin = approx_nonlin
        
        needs_global = interaction in ['mixing', 'adding', 'global_only', 'mixing_adding'] or not approx_nonlin
        needs_local = interaction in ['mixing', 'adding', 'local_only', 'mixing_adding'] or approx_nonlin
        
        if needs_global:
            self.gt = GlobalTransformation(in_dim, out_dim, modes, coor_dim, device)
        if needs_local:
            self.lt = LocalTransformation(in_dim, out_dim, ks, coor_dim, device)
            if interaction == 'mixing_adding':
                self.lt_add = LocalTransformation(in_dim, out_dim, ks, coor_dim, device)
    
    def forward(self, c: torch.Tensor, grid_width: float = 1) -> torch.Tensor:
        if self.approx_nonlin:
            if self.interaction == 'mixing':
                return self.gt(c) * self.lt(c, grid_width)
            elif self.interaction == 'adding':
                return self.gt(c) + self.lt(c, grid_width)
            elif self.interaction == 'mixing_adding':
                return self.gt(c) * self.lt(c, grid_width) + self.lt_add(c, grid_width)
            elif self.interaction == 'local_only':
                return self.lt(c, grid_width)
            elif self.interaction == 'global_only':
                return self.gt(c)
        return self.lt(c, grid_width)


# ==============================================================================
# LGM Layer (Exp 2 & 5)
# ==============================================================================

class LGMLayer(nn.Module):
    """LGM layer with configurable branches and residual operator."""
    
    def __init__(self, in_dim: int, out_dim: int, device, num_nonlin: int, num_lin: int,
                 A_ks: int, coor_dim: int, modes: int, ks: int, interaction: str,
                 use_conv: bool, use_diff: bool, use_nonlin: bool, A_type: str):
        super().__init__()
        self.use_conv, self.use_diff, self.use_nonlin = use_conv, use_diff, use_nonlin
        conv_class = get_conv_class(coor_dim)
        
        # Nonlinear branch
        self.nonlin_layers = nn.ModuleList()
        if use_nonlin:
            for _ in range(num_nonlin):
                self.nonlin_layers.append(LGMTransformation(in_dim, out_dim, device, modes, ks, coor_dim, interaction, True))
        
        # Linear branches
        self.lin_layers = nn.ModuleList()
        for _ in range(num_lin):
            if use_conv:
                self.lin_layers.append(LocalTransformation(in_dim, out_dim, ks, coor_dim, device, False))
            if use_diff:
                self.lin_layers.append(LocalTransformation(in_dim, out_dim, ks, coor_dim, device, True))
        
        # Residual operator A
        self.A_op = None
        if use_nonlin and A_type != 'none':
            if A_type == 'dim_cvc':
                self.A_op = nn.Sequential(
                    make_conv(conv_class, out_dim, 2*out_dim, A_ks, device),
                    make_conv(conv_class, 2*out_dim, out_dim, A_ks, device)
                )
            elif A_type == 'dim_cc':
                self.A_op = make_conv(conv_class, out_dim, out_dim, A_ks, device)
            elif A_type == 'dim_ccc':
                self.A_op = nn.Sequential(
                    make_conv(conv_class, out_dim, out_dim, A_ks, device),
                    make_conv(conv_class, out_dim, out_dim, A_ks, device)
                )
    
    def forward(self, c: torch.Tensor, grid_width: float = 1) -> torch.Tensor:
        out = 0.0
        
        if self.use_nonlin:
            nonlin_sum = sum(layer(c, grid_width) for layer in self.nonlin_layers)
            out = out + (self.A_op(nonlin_sum) if self.A_op else nonlin_sum)
        
        if self.use_conv or self.use_diff:
            out = out + sum(layer(c, grid_width) for layer in self.lin_layers)
        
        return out


# ==============================================================================
# Dynamics Informed Architecture (Exp 3)
# ==============================================================================

class DyIA(nn.Module):
    """DyIA with configurable architecture."""
    
    def __init__(self, in_dim: int, out_dim: int, device, num_layers: int, num_nonlin: int,
                 num_lin: int, A_ks: int, coor_dim: int, modes: int, ks: int,
                 interaction: str, use_conv: bool, use_diff: bool, use_nonlin: bool,
                 A_type: str, arch_type: str, use_gating: bool):
        super().__init__()
        self.num_layers = num_layers
        self.arch_type = arch_type
        self.use_gating = use_gating
        self.out_dim = out_dim
        
        conv_class = get_conv_class(coor_dim)
        
        self.layers = nn.ModuleList([
            LGMLayer(in_dim, out_dim, device, num_nonlin, num_lin, A_ks, coor_dim, 
                     modes, ks, interaction, use_conv, use_diff, use_nonlin, A_type)
            for _ in range(num_layers)
        ])
        
        if arch_type != 'only_sequential':
            self.evo_weights = conv_class(num_layers, 1, 1, 1, 0, padding_mode='circular', device=device)
        
        self.act = nn.GELU()
        self.lambda_ = nn.Parameter(torch.tensor(float(out_dim)))
    
    def forward(self, c: torch.Tensor, temp_coeff: torch.Tensor = None, grid_width: float = 1) -> torch.Tensor:
        c_init = c
        
        if self.arch_type == 'hybrid':
            outputs = []
            for i, layer in enumerate(self.layers):
                F_c = self.act(layer(c, grid_width))
                if i < self.num_layers - 1:
                    c = self._update(c, F_c, temp_coeff)
                outputs.append(F_c)
            return c_init + self._weighted_sum(outputs, c.shape)
            
        elif self.arch_type == 'only_sequential':
            for layer in self.layers:
                F_c = self.act(layer(c, grid_width))
                c = self._update(c, F_c, temp_coeff)
            return c
            
        else:  # only_parallel
            outputs = [self.act(layer(c_init, grid_width)) for layer in self.layers]
            return c_init + self._weighted_sum(outputs, c.shape)
    
    def _update(self, c, F_c, temp_coeff):
        if self.use_gating:
            eff = self.lambda_ * temp_coeff if temp_coeff is not None else self.lambda_
            return c + F_c * torch.sigmoid(eff)
        return c + F_c
    
    def _weighted_sum(self, outputs, shape):
        stacked = torch.stack(outputs, dim=2)
        view = stacked.view(-1, self.num_layers, *shape[2:])
        return self.evo_weights(view).view(shape[0], -1, *shape[2:])


# ==============================================================================
# Main Model
# ==============================================================================

class DyMixOp(nn.Module):
    """DyMixOp ablation model."""
    
    def __init__(self, config, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device
        
        self.input_steps = 1 + config.inp_involve_history_step
        self.seq_len = config.ar_nseq_train + config.ar_nseq_test
        self.input_ch = config.input_dim * self.input_steps + config.coor_input_dim
        
        v_dim = config.hidden_dim * 2
        c_dim = config.hidden_dim
        
        conv_class = get_conv_class(config.coor_input_dim)
        ks = getattr(config, 'kernel_size', 1)
        self.num_nonlinear = getattr(config, 'num_nonlinear', 1)
        self.num_linear = getattr(config, 'num_linear', 1)
        self.A_op_ks = getattr(config, 'A_op_ks', 1)
        self.infer_factor = getattr(config, 'infer_scaling_factor', 1.0)
        
        # Build lifting/projection based on dimension reduction type
        dim_type = config.dimension_reduction_type
        
        if dim_type == 'full_reduction':
            self.lift = make_conv(conv_class, self.input_ch, v_dim, ks, device)
            self.encode = make_conv(conv_class, v_dim, c_dim, ks, device)
            
        elif dim_type == 'cpx_reduc_dim_vvcc':
            self.lift = make_conv_stack(conv_class, [self.input_ch, v_dim, v_dim], ks, device)
            self.encode = make_conv_stack(conv_class, [v_dim, c_dim, c_dim], ks, device)
            
        else:  # no_projection
            v_dim = config.hidden_dim
            self.lift = make_conv_stack(conv_class, [self.input_ch, v_dim, v_dim], ks, device)
            self.encode = nn.Identity()
        
        self.c_dim = c_dim if dim_type != 'no_projection' else v_dim
        
        self.dyia = DyIA(
            self.c_dim, self.c_dim, device, config.num_layers,
            self.num_nonlinear, self.num_linear, self.A_op_ks,
            config.coor_input_dim, config.global_modes, ks,
            config.interaction_type, config.use_conv_linear_branch,
            config.use_diff_linear_branch, config.use_nonlinear_branch,
            config.residual_operator_A_type, config.architecture_type,
            config.use_adaptive_gating
        )

        if dim_type == 'full_reduction':
            self.decode = make_conv(conv_class, c_dim, v_dim, ks, device)
            self.project = make_conv(conv_class, v_dim, config.output_dim, ks, device)
            
        elif dim_type == 'cpx_reduc_dim_vvcc':
            self.decode = make_conv_stack(conv_class, [c_dim, c_dim, v_dim], ks, device)
            self.project = make_conv_stack(conv_class, [v_dim, v_dim, config.output_dim], ks, device)
            
        else:  # no_projection
            v_dim = config.hidden_dim
            self.decode = nn.Identity()
            self.project = make_conv_stack(conv_class, [v_dim, v_dim, config.output_dim], ks, device)
        
    
    def forward(self, x: torch.Tensor, static_data: List[torch.Tensor], grid_width: float = 1) -> torch.Tensor:
        coor = static_data[0]
        temp_coeff = static_data[1][0] if len(static_data) > 1 else None
        
        outputs = []
        for _ in range(self.seq_len):
            x_pred = self._step(torch.cat((x, coor), dim=1), temp_coeff, grid_width / self.infer_factor)
            outputs.append(x_pred)
            
            if self.input_steps > 1:
                x = torch.cat((x[:, self.config.input_dim:], x_pred), dim=1)
            else:
                x = x_pred
        
        return torch.stack(outputs, dim=1)
    
    def _step(self, u: torch.Tensor, temp_coeff, grid_width) -> torch.Tensor:
        v = self.lift(u)
        c = self.encode(v)
        c = self.dyia(c, temp_coeff, grid_width)
        v = self.decode(c)
        return self.project(v)
