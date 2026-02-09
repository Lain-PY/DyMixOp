import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple
import torch.nn.functional as F
import math
import numpy as np
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict


## note
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from Models.LaMO_utils import cross_scan_fn, cross_merge_fn

# ==============================================================================
# Helper Modules (Mamba Core, MLP, Attention, Patchify)
# ==============================================================================

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)

class mamba_4D(nn.Module):
    def __init__(
        self,
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.GELU,
        d_conv=3,
        conv_bias=True,
        dropout=0.0,
        bias=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v2",
        forward_type="m0",
        with_initial_state=False,
        force_fp32=False,
        chunk_size=64,
        selective_scan_backend=None,
        scan_mode="cross2d",
        scan_force_torch=False,
        **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        
        self.with_dconv = d_conv > 1
        self.d_state = d_state
        self.force_fp32 = force_fp32
        self.chunk_size = chunk_size
        Linear = nn.Linear
        
        self.disable_force32 = False
        self.oact = False
        self.disable_z = False
        self.disable_z_act = False
        self.out_norm = nn.LayerNorm(d_inner)
        
        self.selective_scan_backend = selective_scan_backend
        self.scan_mode = scan_mode
        self.scan_force_torch = scan_force_torch

        k_group = 4

        # in proj
        d_proj = d_inner if self.disable_z else (d_inner * 2)
        self.in_proj = Linear(d_model, d_proj, bias=bias)
        self.act: nn.Module = act_layer()
        
        # conv
        if self.with_dconv:
            self.conv2d = nn.Sequential(
                Permute(0, 3, 1, 2),
                nn.Conv2d(
                    in_channels=d_inner,
                    out_channels=d_inner,
                    groups=d_inner,
                    bias=conv_bias,
                    kernel_size=d_conv,
                    padding=(d_conv - 1) // 2,
                    **factory_kwargs,
                ),
                Permute(0, 2, 3, 1),
            ) 
        
        # x proj
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj
        
        # out proj
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v1"]:
            self.Ds = nn.Parameter(torch.ones((k_group, dt_rank, int(d_inner // dt_rank))))
            self.A_logs = nn.Parameter(torch.randn((k_group, dt_rank))) 
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((k_group, dt_rank)))
        elif initialize in ["v2"]:
            self.Ds = nn.Parameter(torch.ones((k_group, dt_rank, int(d_inner // dt_rank))))
            self.A_logs = nn.Parameter(torch.zeros((k_group, dt_rank))) 
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, dt_rank)))

        # init state
        self.initial_state = None
        if with_initial_state:
            self.initial_state = nn.Parameter(torch.zeros((1, k_group * dt_rank, int(d_inner // dt_rank), d_state)), requires_grad=False)

    def forward_core(self, x: torch.Tensor=None, **kwargs):
        x_proj_bias = getattr(self, "x_proj_bias", None)
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        N = self.d_state
        B, H, W, RD = x.shape
        K, R = self.A_logs.shape
        K, R, D = self.Ds.shape
        
        L = H * W
        KR = K * R
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=3)[self.scan_mode]

        initial_state = None
        if self.initial_state is not None:
            initial_state = self.initial_state.detach().repeat(B, 1, 1, 1)
            
        xs = cross_scan_fn(x.view(B, H, W, RD), in_channel_first=False, out_channel_first=False, scans=_scan_mode, force_torch=self.scan_force_torch)
        
        x_dbl = torch.einsum("b l k d, k c d -> b l k c", xs, self.x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, -1, K, 1)
        
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=3)
        xs = xs.contiguous().view(B, L, KR, D)
        dts = dts.contiguous().view(B, L, KR)
        Bs = Bs.contiguous().view(B, L, K, N)
        Cs = Cs.contiguous().view(B, L, K, N)
        
        if self.force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        As = -self.A_logs.to(torch.float).exp().view(KR)
        Ds = self.Ds.to(torch.float).view(KR, D)
        dt_bias = self.dt_projs_bias.view(KR)

        if self.force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)
        
        ys, final_state = mamba_chunk_scan_combined(
            xs, dts, As, Bs, Cs, chunk_size = self.chunk_size, D=Ds, dt_bias=dt_bias, 
            initial_states=initial_state, dt_softplus=True, return_final_states=True,
        )
                
        y: torch.Tensor = cross_merge_fn(ys.view(B, H, W, K, RD), in_channel_first=False, out_channel_first=False, scans=_scan_mode, force_torch=self.scan_force_torch)
            
        if self.initial_state is not None:
            self.initial_state = nn.Parameter(final_state.detach().sum(0, keepdim=True), requires_grad=False)

        y = self.out_norm(y.view(B, H, W, -1))
        return y.to(x.dtype)

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=(-1)) 
            if not self.disable_z_act:
                z = self.act(z)
        if self.with_dconv:
            x = self.conv2d(x)
        x = self.act(x)
        y = self.forward_core(x)
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Patchify(nn.Module):
    def __init__(self, H_img=224, W_img=224, patch_size=3, in_chans=3, embed_dim=768):
        super(Patchify, self).__init__()
        assert H_img % patch_size == 0
        assert W_img % patch_size == 0
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.H, self.W = H_img // patch_size[0], W_img // patch_size[1]
        self.num_patches = self.H * self.W
        self.WE = nn.Parameter(torch.randn(self.embed_dim, self.in_chans, patch_size[0], patch_size[1]))
        self.bE = nn.Parameter(torch.randn(self.embed_dim))
        self.layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.unfold(2, self.patch_size[0], self.patch_size[0]).unfold(3, self.patch_size[0], self.patch_size[1])
        x = x.contiguous().view(B, self.in_chans, self.H * self.W, self.patch_size[0], self.patch_size[1])
        embedded_tokens = torch.einsum('ijklm,cjlm->ick', x, self.WE) + self.bE.view(1, -1, 1)
        embedded_tokens = embedded_tokens.permute(0, 2, 1).contiguous()
        embedded_tokens = self.layer_norm(embedded_tokens)
        return embedded_tokens    

class DePatchify(nn.Module):
    def __init__(self, H_img=224, W_img=224, patch_size=3, embed_dim=3, in_chans=768):
        super(DePatchify, self).__init__()
        assert H_img % patch_size == 0
        assert W_img % patch_size == 0
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.H, self.W = H_img // patch_size[0], W_img // patch_size[1]
        self.num_patches = self.H * self.W
        self.WE = nn.Parameter(torch.randn(self.embed_dim, self.in_chans, patch_size[0], patch_size[1]))
        self.bE = nn.Parameter(torch.randn(self.embed_dim))
        self.layer_norm = nn.LayerNorm(self.embed_dim)
    
    def forward(self, x):
        x = self.layer_norm(x)
        y = x + self.bE.reshape(1, 1, -1)
        y = torch.einsum('ijk,klmn->iljmn', y, self.WE)
        y = y.reshape(y.shape[0], y.shape[1], self.H, self.W, y.shape[3], y.shape[4]).permute(0, 1, 2, 4, 3, 5)
        y = y.reshape(y.shape[0], y.shape[1], y.shape[2] * y.shape[3], y.shape[4] * y.shape[5])
        return y

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, H=85, W=85):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.hydra = mamba_4D(d_model=dim, d_state=64, ssm_ratio=2.0)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.H = H
        self.W = W
        self.dim = dim
    
    def forward(self, x):
        inp = x
        outp = self.drop_path(self.hydra(self.norm1(x).reshape(-1, self.H, self.W, self.dim)).reshape(-1, self.H*self.W, self.dim))
        x = inp + outp
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, inp, outp

# ==============================================================================
# LaMO Model Wrapper (Adapted for Framework Interface)
# ==============================================================================

class LaMO(nn.Module):
    def __init__(self, model_config, device='cuda'):
        super(LaMO, self).__init__()
        
        # 1. Extract parameters from model_config
        # We try to find specific LaMO keys, otherwise we map from generic keys
        self.space_dim = getattr(model_config, 'coor_input_dim', 2)
        
        # Input/Output dimensions
        # We need to handle time history accumulation
        self.involve_history_step = getattr(model_config, 'inp_involve_history_step', 0) # e.g., 0 means use current only
        self.input_fun_dim = getattr(model_config, 'input_dim', 1) 
        self.output_dim = getattr(model_config, 'output_dim', 1)
        self.seq_len = getattr(model_config, 'ar_nseq_train', 1) + getattr(model_config, 'ar_nseq_test', 0) # Just for info
        
        # LaMO specific: Total function channels = (history + 1) * fun_dim
        self.total_fun_dim = self.input_fun_dim * (self.involve_history_step + 1)
        
        # Resolution
        # If not provided in config, default to 85 (from original LaMO) or 64
        self.H_img = getattr(model_config, 'H_img', getattr(model_config, 'resolution', 64))
        self.W_img = getattr(model_config, 'W_img', getattr(model_config, 'resolution', 64))
        
        # Padding dims (usually slightly larger or divisible by patch size)
        # Here we default to slightly larger multiple of max patch size if not specified
        self.H_pad = getattr(model_config, 'H_pad', 64) 
        self.W_pad = getattr(model_config, 'W_pad', 64)

        # Architecture hyperparams
        self.embed_dims = getattr(model_config, 'hidden_dim', [256])
        self.num_heads = getattr(model_config, 'n_heads', [8])
        self.mlp_ratios = getattr(model_config, 'mlp_ratios', [4])
        self.depths = getattr(model_config, 'num_layers', [6])
        self.patch_sizes = getattr(model_config, 'patch_sizes', [2])
        self.num_scales = len(self.embed_dims)
        
        drop_rate = getattr(model_config, 'dropout', 0.0)
        attn_drop_rate = getattr(model_config, 'attn_dropout', 0.0)
        drop_path_rate = getattr(model_config, 'drop_path', 0.0)
        
        # 2. Build Layers
        for i in range(self.num_scales):
            # Input channels for first scale includes space_dim + function_dim
            # Subsequent scales take embedding from previous
            in_chans = (self.total_fun_dim + self.space_dim) if i == 0 else self.embed_dims[i-1]
            
            patch_embed = Patchify(
                H_img=self.H_pad,
                W_img=self.W_pad,
                patch_size=self.patch_sizes[i],
                in_chans=in_chans,
                embed_dim=self.embed_dims[i]
            )
            
            num_patches = (self.H_pad // self.patch_sizes[i]) * (self.W_pad // self.patch_sizes[i])
            pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dims[i]))
            pos_drop = nn.Dropout(p=drop_rate)
            
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
            cur = sum(self.depths[:i])
            
            block = nn.ModuleList([Block(
                dim=self.embed_dims[i], 
                num_heads=self.num_heads[i], 
                mlp_ratio=self.mlp_ratios[i], 
                drop=drop_rate, 
                attn_drop=attn_drop_rate, 
                drop_path=dpr[cur + j],
                H=self.H_pad // self.patch_sizes[i], 
                W=self.W_pad // self.patch_sizes[i]
            ) for j in range(self.depths[i])])
            
            # Output channels: if last scale, project to output_dim, else project to next embed_dim
            # out_chans_layer = self.output_dim * self.seq_len if i == self.num_scales - 1 else self.embed_dims[i] # autoregressive
            out_chans_layer = self.output_dim * 1 if i == self.num_scales - 1 else self.embed_dims[i] # non-autoregressive
            
            depatch_embed = DePatchify(
                H_img=self.H_pad,
                W_img=self.W_pad,
                patch_size=self.patch_sizes[i],
                embed_dim=self.embed_dims[i],
                in_chans=out_chans_layer
            )
             
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"pos_embed{i + 1}", pos_embed)
            setattr(self, f"pos_drop{i + 1}", pos_drop)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"depatch_embed{i+1}", depatch_embed)

        self.placeholder = nn.Parameter((1 / (self.space_dim)) * torch.rand(self.space_dim, dtype=torch.float))
        
        self.to(device)

    def forward_features(self, x):
        features = []
        outs = []
        # x shape at start: [B, C, H, W]
        
        for i in range(self.num_scales):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            pos_drop = getattr(self, f"pos_drop{i + 1}")
            block = getattr(self, f"block{i + 1}")
            depatch_embed = getattr(self, f"depatch_embed{i + 1}")

            # Patchify
            x = patch_embed(x) # [B, N_patches, Embed_dim]
            x = pos_drop(x + pos_embed)
            
            # Process Blocks
            for blk in block:
                x, inp, outp = blk(x)
                features.append(inp)
                features.append(outp)

            # DePatchify
            x = depatch_embed(x) # [B, Out_C or Next_Embed, H, W]
            outs.append(x)
            
        return outs, features

    def model_forward(self, x, fx):
        """
        Internal forward logic adapted from original LaMO.
        x: Coordinates [Batch*Pixels, Space_Dim] or [Batch, H, W, Space_Dim]
        fx: Function values [Batch*Pixels, Fun_Dim] or [Batch, H, W, Fun_Dim]
        """
        # Ensure inputs are shaped [Batch, H, W, Channels] for concatenation
        if x.dim() == 2:
            # Assume flat input, try to reshape
            B = x.shape[0] // (self.H_img * self.W_img)
            x = x.reshape(B, self.H_img, self.W_img, self.space_dim)
        
        if fx is not None:
            if fx.dim() == 2:
                B = fx.shape[0] // (self.H_img * self.W_img)
                fx = fx.reshape(B, self.H_img, self.W_img, -1)
            # Concatenate coordinates and function values along channel dimension
            # Result: [Batch, H, W, Space_Dim + Fun_Dim]
            combined = torch.cat((x, fx), -1)
        else:
            # Use placeholder if no function input (unlikely in PDE solving)
            fx = x + self.placeholder[None, None, :]
            combined = fx

        # Permute to [Batch, Channels, H, W] for Patchify
        combined = combined.permute(0, 3, 1, 2)
      
        # Padding
        pad_h = self.H_pad - self.H_img  
        pad_w = self.W_pad - self.W_img 
        if pad_h > 0 or pad_w > 0:
            combined = F.pad(combined, (0, pad_w, 0, pad_h), mode='constant', value=0)
        
        # Backbone
        outs, features = self.forward_features(combined)
        final_out = outs[-1] # [Batch, Out_Dim, H_pad, W_pad]
        
        # Unpadding
        if pad_h > 0 or pad_w > 0:
            final_out = final_out[:, :, :self.H_img, :self.W_img]
            
        return final_out

    def forward(self, input_data, static_data):
        """
        New interface for DyMixOp compatibility with autoregressive functionality.
        
        Args:
            input_data: Temporal snapshots. Shape [Batch, Time*Fun_Dim, H, W]
            static_data: Static info (Coordinates). Shape [Batch, Space_Dim, H*W] or similar.
                        Usually static_data[0] is the grid.
        
        Returns:
            x_out: Snapshots at next time step.
        """
        
        x_list = []
        x_in = input_data
        coor_input = static_data[0]
        
        # Handle Grid (x)
        batch_size = x_in.shape[0]
        grid = coor_input  # Typically [Batch, Space_Dim, Spatial_Nodes]
        
        # 1. Check if spatial dim is 1 and then extend it to 2D
        if input_data.dim() == 3:
            grid = grid.unsqueeze(-1)
            x_in = x_in.unsqueeze(-1)

        # [B, C, H, W] -> [B, H, W, C]
        grid = grid.permute(0, 2, 3, 1)
        
        for k in range(self.seq_len):
            u_p = x_in.permute(0, 2, 3, 1)

            # 2. Run Model
            # model_forward expects x:[B, H, W, Space_dim], fx:[B, H, W, T*C]
            # output is [B, Out_Dim, H, W]
            out_tensor = self.model_forward(grid, u_p)
            
            # Add to list of predictions
            x_list.append(out_tensor)
            
            # Prepare input for next iteration
            # Concatenate the latest prediction with the input (autoregressive)
            # Remove the oldest time step and add the new prediction
            if k < self.seq_len - 1:  # Not the last iteration
                x_in = torch.cat((x_in[:, self.input_fun_dim:], out_tensor), dim=1)

        # Stack all predictions along the time dimension
        x = torch.stack(x_list, dim=1)

        if input_data.dim() == 3:
            # Squeeze spatial dims
            x = x.squeeze(-1)
        
        return x
