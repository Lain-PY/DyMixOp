# Configuration Directory Structure

This directory contains configuration files organized by **baseline model** and **purpose**.

## Directory Schema

```
Configs/
├── {baseline}/           # e.g., DyMixOp, FNO, UNO, LocalNO
│   ├── Base/             # Canonical base configurations
│   ├── Variants/         # Scaled/modified configurations (different sizes, hyperparameters)
│   └── Ablation/         # Ablation study configurations
└── README.md
```

## Purpose Definitions

| Purpose | Description |
|---------|-------------|
| **Base** | Canonical configuration files used as starting points. These are complete, valid configurations for training/inference. |
| **Variants** | Derived configurations with different architecture sizes (Tiny/Medium/Large), training budgets, or capacity settings. Generated from base configs. |
| **Ablation** | Modified configurations for ablation experiments (e.g., removing components, altering connectivity). |

## File Naming Convention

```
config_{benchmark}_{baseline}.json
```

**Examples:**

- `config_2dburgers_DyMixOp.json` - 2D Burgers equation benchmark with DyMixOp
- `config_3dsw_FNO.json` - 3D Shallow Water benchmark with FNO

**Full Path Pattern:**

```
Configs/{baseline}/{purpose}/config_{benchmark}_{baseline}.json
```

## Supported Benchmarks

| Benchmark | Description |
|-----------|-------------|
| `1dks` | 1D Kuramoto-Sivashinsky equation |
| `2dburgers` | 2D Burgers equation |
| `2dce-crp` | 2D Compressible Euler (CRP) |
| `2ddarcy` | 2D Darcy flow |
| `2dns` | 2D Navier-Stokes |
| `3dbrusselator` | 3D Brusselator reaction-diffusion |
| `3dsw` | 3D Shallow Water equations |

## Supported Baselines

| Baseline | Model Path | Description |
|----------|-----------|-------------|
| `DyMixOp` | `./Models/DyMixOp.py` | Dynamic Mixing Operator (our method) - Hybrid local-global operator with sequential-parallel architecture |
| `FNO` | `./Models/FNO.py` | Fourier Neural Operator |
| `UNO` | `./Models/UNO.py` | U-shaped Neural Operator |
| `LocalNO` | `./Models/LocalNO.py` | Local Neural Operator |
| `GNOT` | `./Models/GNOT.py` | Graph Neural Operator Transformer |
| `LaMO` | `./Models/LaMO.py` | Latent Mixing Operator |
| `DeepONet` | `./Models/DeepONet.py` | Deep Operator Network |
| `CoDANO` | `./Models/CoDANO.py` | Convolutional Deep Attentive Neural Operator |
| `ConvLSTM` | `./Models/ConvLSTM.py` | Convolutional LSTM |

## DyMixOp Model Parameters

The following parameters are specific to DyMixOp configurations:

### Architecture Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | string | Required | Model identifier (e.g., `"DyMixOp"`) |
| `model_path` | string | Required | Path to model file (e.g., `"./Models/DyMixOp.py"`) |
| `hidden_dim` | int | Required | Hidden dimension size |
| `num_layers` | int | Required | Number of DyIA layers |
| `global_modes` | int | Required | Number of Fourier modes for global transformation |
| `kernel_size` | int | 1 | Convolution kernel size for local transformations |
| `num_nonlinear` | int | 1 | Number of nonlinear (LGM) branches per layer |
| `num_linear` | int | 1 | Number of linear (conv + diff) branches per layer |
| `A_op_ks` | int | 1 | Kernel size for residual operator A |

### Input/Output Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_dim` | int | Required | Input field dimension (channels) |
| `output_dim` | int | Required | Output field dimension (channels) |
| `coor_input_dim` | int | Required | Coordinate dimension (1D/2D/3D) |
| `inp_involve_history_step` | int | Required | Number of historical steps to include in input |
| `ar_nseq_train` | int | Required | Autoregressive sequence length during training |
| `ar_nseq_test` | int | Required | Autoregressive sequence length during testing |

### Mesh Invariance Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `infer_scaling_factor` | float | 1.0 | Scaling factor for zero-shot super-resolution inference. Set to match spatial resolution ratio (e.g., 2.0 for 2× resolution) |
| `train_shape` | list | Required | Spatial resolution for training (e.g., `[256]` for 1D, `[64, 64]` for 2D) |

> **Note**: When performing scaled inference at different resolutions, set `infer_scaling_factor` to the ratio between inference and training resolutions. This enables mesh-invariant predictions via proper grid width adjustment in differential operators.

## Usage in Scripts

The configuration path is resolved using:

```python
CONFIG_PATH_PATTERN = "Configs/{baseline}/{purpose}/config_{benchmark}_{baseline}.json"

# Example: resolve_config_path("2dBurgers", "DyMixOp", purpose="Base")
# Returns: "Configs/DyMixOp/Base/config_2dburgers_DyMixOp.json"
```

## Example Configuration Structure

```json
{
  "model": {
    "model_name": "DyMixOp",
    "model_path": "./Models/DyMixOp.py",
    "input_dim": 1,
    "output_dim": 1,
    "coor_input_dim": 1,
    "hidden_dim": 16,
    "num_layers": 2,
    "global_modes": 12,
    "kernel_size": 1,
    "num_nonlinear": 1,
    "num_linear": 1,
    "A_op_ks": 1,
    "train_shape": [256],
    "infer_scaling_factor": 1.0
  }
}
```
