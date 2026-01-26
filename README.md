# JAX N-Body Emulator

A JAX-based neural network emulator for N-body cosmological simulations.

Based on the PyTorch version trained with map2map:
- https://github.com/eelregit/map2map
- https://github.com/dsjamieson/map2map_emu
- https://github.com/dsjamieson/map2map_zdep

We kindly request to cite these papers in scientific applications of this code:
- [arXiv:2408.07699](https://arxiv.org/abs/2408.07699)
- [arXiv:2206.04594](https://arxiv.org/abs/2206.04594)
- [arXiv:2206.04573](https://arxiv.org/abs/2206.04573)

## Installation

### Basic Installation (CPU)

```bash
pip install jax_nbody_emulator
```

This installs the package with CPU-only JAX support.

### GPU Installation

For GPU support, first install JAX with CUDA support, then install the emulator.

CUDA 12:

```bash
pip install --upgrade "jax[cuda12]"
pip install jax_nbody_emulator
```

CUDA 11:

```bash
pip install --upgrade "jax[cuda11]"
pip install jax_nbody_emulator
```

See the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for more details.

### From Source (Editable Installation)

```bash
git clone https://github.com/dsjamieson/jax_nbody_emulator.git
cd jax_nbody_emulator
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/dsjamieson/jax_nbody_emulator.git
cd jax_nbody_emulator
pip install -e ".[dev]"
```

## Quick Start

### Recommended: Using `create_emulator`

The simplest way to use the emulator is with the `create_emulator` factory function:

```python
import numpy as np
from jax_nbody_emulator import create_emulator, SubboxConfig

# Configure subbox processing
config = SubboxConfig(
    size=(512, 512, 512),      # Full box size
    ndiv=(4, 4, 4),            # Divide into 64 subboxes
)

# Create emulator with velocity output
emulator = create_emulator(
    compute_vel=True,          # Return both displacement and velocity
    processor_config=config,
)

# Load your input displacement field (3, 512, 512, 512)
input_displacement = np.random.randn(3, 512, 512, 512).astype(np.float32)

# Process the full box
displacement, velocity = emulator.process_box(
    input_displacement,
    z=0.5,
    Om=0.3,
)

print(f"Output displacement shape: {displacement.shape}")  # (3, 512, 512, 512)
print(f"Output velocity shape: {velocity.shape}")          # (3, 512, 512, 512)
```

### Fixed Cosmology Mode (Faster for Repeated Inference)

If you're processing many boxes with the same cosmology, premodulate the parameters for faster inference:

```python
from jax_nbody_emulator import create_emulator, SubboxConfig

config = SubboxConfig(
    size=(512, 512, 512),
    ndiv=(4, 4, 4),
)

# Premodulate parameters for fixed (z, Om)
emulator = create_emulator(
    premodulate=True,
    premodulate_z=0.5,
    premodulate_Om=0.3,
    compute_vel=True,
    processor_config=config,
)

# Process multiple boxes with same cosmology
for input_box in input_boxes:
    displacement, velocity = emulator.process_box(input_box, z=0.5, Om=0.3)
```

### Direct Model Access

For more control, you can use the model classes directly:

```python
import jax
import jax.numpy as jnp
from jax_nbody_emulator import (
    StyleNBodyEmulatorVelCore,
    load_default_parameters,
    growth_factor,
    vel_norm,
)

# Load pretrained parameters
params = load_default_parameters()

# Create model
model = StyleNBodyEmulatorVelCore()

# Prepare input (batch, channels, D, H, W)
# Input size must be output_size + 2 * padding (e.g., 32 + 96 = 128)
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (1, 3, 128, 128, 128))

# Cosmological parameters
z = 0.5
Om = 0.3
Dz = growth_factor(jnp.array([z]), jnp.array([Om]))
vn = vel_norm(jnp.array([z]), jnp.array([Om]))

# Run inference
displacement, velocity = model.apply(params, x, jnp.array([Om]), Dz, vn)

print(f"Output shape: {displacement.shape}")  # (1, 3, 32, 32, 32)
```

### Cosmology Functions

The package provides cosmological functions for flat ΛCDM:

```python
from jax_nbody_emulator import (
    growth_factor,    # Linear growth factor D(z), normalized to 1 at z=0
    hubble_rate,      # Hubble parameter H(z) in units of h km/s/Mpc
    growth_rate,      # Linear growth rate f = d ln D / d ln a
    vel_norm,         # Velocity normalization factor
    acc_norm,         # Acceleration normalization factor
)

import jax.numpy as jnp

z = jnp.array([0.0, 0.5, 1.0])
Om = jnp.array([0.3, 0.3, 0.3])

D = growth_factor(z, Om)   # [1.0, 0.77, 0.61]
H = hubble_rate(z, Om)     # [100, 131, 176] h km/s/Mpc
f = growth_rate(z, Om)     # [0.51, 0.75, 0.87]
```

## API Reference

### Primary API

| Function/Class | Description |
|----------------|-------------|
| `create_emulator()` | Factory function to create configured emulator |
| `NBodyEmulator` | Emulator bundle with model, params, and processor |
| `SubboxConfig` | Configuration for subbox processing |
| `SubboxProcessor` | Processes large volumes via subbox decomposition |
| `load_default_parameters()` | Load pretrained model parameters |
| `modulate_emulator_parameters()` | Premodulate params for fixed cosmology |
| `modulate_emulator_parameters_vel()` | Premodulate params with velocity support |

### Core Models

| Class | Description |
|-------|-------------|
| `StyleNBodyEmulatorCore` | Style model (flexible cosmology) |
| `StyleNBodyEmulatorVelCore` | Style model with velocity output |
| `NBodyEmulatorCore` | Premodulated model (fixed cosmology) |
| `NBodyEmulatorVelCore` | Premodulated model with velocity output |

### Cosmology Functions

| Function | Description |
|----------|-------------|
| `growth_factor(z, Om)` | Linear growth factor D(z) |
| `hubble_rate(z, Om)` | Hubble parameter H(z) |
| `growth_rate(z, Om)` | Growth rate f(z) |
| `vel_norm(z, Om)` | Velocity normalization |
| `acc_norm(z, Om)` | Acceleration normalization |

## Features

- **JAX-accelerated** cosmological calculations and neural network inference
- **Automatic differentiation** for velocities via forward-mode AD
- **StyleGAN-inspired** 3D U-Net architecture with cosmology conditioning
- **Efficient subbox processing** for arbitrarily large volumes
- **FP16/FP32 precision** support for memory/speed tradeoffs
- **Modular design** with layers, blocks, and full models
- **Comprehensive test suite** with 500+ tests

## Requirements

- Python ≥3.10
- JAX ≥0.4.0
- Flax ≥0.7.0
- NumPy ≥1.21.0
- tqdm ≥4.60.0

GPU Requirements:
- NVIDIA GPU with CUDA support
- CUDA 11.8+ or 12.0+
- 15-40 GB GPU memory (depends on subbox size and precision)

## Performance

Benchmarks for processing a 512³ volume on NVIDIA A100 (40GB), excluding initial compilation:

| Precision | Premodulate | Velocity | Subbox Size | Num Subboxes | Time |
|-----------|-------------|----------|-------------|--------------|------|
| FP16 | Yes | No | 128×256×256 | 16 | 10.9s |
| FP16 | No | No | 128×256×256 | 16 | 11.1s |
| FP32 | Yes | No | 128×128×256 | 32 | 15.6s |
| FP32 | No | No | 128×128×256 | 32 | 15.8s |
| FP16 | Yes | Yes | 128×256×256 | 16 | 25.8s |
| FP16 | No | Yes | 128×256×256 | 16 | 25.9s |
| FP32 | Yes | Yes | 128×128×128 | 64 | 44.7s |
| FP32 | No | Yes | 128×128×128 | 64 | 44.9s |

**Key observations:**
- Velocity computation adds ~2.3× overhead
- FP16 allows larger subboxes, reducing overhead from fewer crops
- Premodulation provides negligible speedup (~1%)
- Optimal FP16 subbox size: 128×256×256 (16 subboxes for 512³ volume)
- Optimal FP32 subbox size: 128×128×256 (no velocity) or 128×128×128 (with velocity)

## Citation

If you use this code in your research, please cite:

```bibtex
@software{jamieson2025jax_nbody,
  author = {Jamieson, Drew},
  title = {JAX N-body Emulator},
  year = {2025},
  url = {https://github.com/dsjamieson/jax_nbody_emulator}
}
```

And the original papers:
- [arXiv:2408.07699](https://arxiv.org/abs/2408.07699)
- [arXiv:2206.04594](https://arxiv.org/abs/2206.04594)
- [arXiv:2206.04573](https://arxiv.org/abs/2206.04573)

## License

GNU General Public License v3.0