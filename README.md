# JAX N-Body Emulator

A JAX-based neural network emulator for N-body cosmological simulations.

Based on the PyTorch version trained with map2map:
- https://github.com/eelregit/map2map
- https://github.com/dsjamieson/map2map_emu

We kindly request to cite these papers in scientific applications of this code:
- arXiv:2408.07699
- arXiv:2206.04594
- arXiv:2206.04573

## Installation

### Basic Installation (CPU)

    pip install jax_nbody_emulator

This installs the package with CPU-only JAX support.

### GPU Installation

For GPU support, first install JAX with CUDA support, then install the emulator.

CUDA 12:

    pip install --upgrade "jax[cuda12]"
    pip install jax_nbody_emulator

CUDA 11:

    pip install --upgrade "jax[cuda11]"
    pip install jax_nbody_emulator

See the JAX installation guide for more details: https://jax.readthedocs.io/en/latest/installation.html

### From Source (Editable Installation)

    git clone https://github.com/dsjamieson/jax_nbody_emulator.git
    cd jax_nbody_emulator
    pip install -e .

### Development Installation

    git clone https://github.com/dsjamieson/jax_nbody_emulator.git
    cd jax_nbody_emulator
    pip install -e ".[dev]"

## Quick Start

### Single Subbox Example

    import jax
    import jax.numpy as jnp
    from jax_nbody_emulator import NBodyEmulatorVel, load_default_parameters
    
    # Use FP16 for faster inference
    emulator_dtype = jnp.float16
    
    # Load pretrained parameters
    data = load_default_parameters(dtype=emulator_dtype)
    params = data['params']
    
    # Create emulator with velocity output
    emulator = NBodyEmulatorVel(dtype=emulator_dtype)
    
    # Create test input
    key = jax.random.PRNGKey(0)
    pad = 48  # Model requires 48 voxels of periodic padding
    ngrid_in = 32  # Size without padding
    ngrid_pad = ngrid_in + 2 * pad  # Size with padding
    input_shape = (1, 3, ngrid_pad, ngrid_pad, ngrid_pad)  # (batch, channels, D, H, W)
    dis_in = jax.random.normal(key, input_shape, dtype=emulator_dtype)
    
    # Set cosmological parameters
    omega_m = jnp.array([0.3], dtype=emulator_dtype)
    redshift = 0.5
    
    # Compute cosmological quantities
    from jax_nbody_emulator import D, vel_norm
    growth_factor = D(jnp.array([redshift]), omega_m)
    vel_factor = vel_norm(jnp.array([redshift]), omega_m)
    
    # Run emulator
    dis_out, vel_out = emulator.apply(params, dis_in, omega_m, growth_factor, vel_factor)
    
    print(f"Output displacement shape: {dis_out.shape}")
    print(f"Output velocity shape: {vel_out.shape}")

### Large Volume Processing with Subboxes

    import numpy as np
    import jax.numpy as jnp
    from jax_nbody_emulator import (
        NBodyEmulatorVel,
        SubboxConfig,
        SubboxProcessorVel,
        load_default_parameters
    )
    
    # Load model with FP16 precision
    emulator_dtype = jnp.float16
    data = load_default_parameters(dtype=emulator_dtype)
    params = data['params']
    model = NBodyEmulatorVel(dtype=emulator_dtype)
    
    # Configure subbox processing for large volumes
    config = SubboxConfig(
        in_chan=3,
        size=(512, 512, 512),  # Full box size
        ndiv=(4, 2, 2),  # Divide into 16 subboxes
        padding=((48, 48), (48, 48), (48, 48))  # Required padding for model
    )
    
    # Create processor
    processor = SubboxProcessorVel(
        model=model,
        params=params,
        config=config,
        dtype=emulator_dtype
    )
    
    # Load your input displacement field (3, 512, 512, 512)
    input_displacement = np.random.randn(3, 512, 512, 512).astype(np.float32)
    
    # Process the full box
    displacement, velocity = processor.process_box(
        input_box=input_displacement,
        z=0.5,
        Om=0.3,
        desc="Processing z=0.5"
    )
    
    print(f"Output displacement shape: {displacement.shape}")
    print(f"Output velocity shape: {velocity.shape}")

## Features

- JAX-accelerated cosmological calculations
- Explicit forward-mode derivatives for velocities
- StyleGAN-inspired 3D U-Net architecture
- Efficient subbox processing for large volumes
- FP16/FP32 precision support
- Modular design for easy extension
- Full type hints and documentation

## Requirements

- Python >=3.8
- JAX >=0.4.0
- Flax >=0.7.0
- NumPy >=1.21.0
- tqdm >=4.60.0

GPU Requirements:
- NVIDIA GPU with CUDA support
- CUDA 11.8+ or 12.0+
- Approximately 15-40 GB GPU memory (depends on box size and precision)

## Performance

- FP16: 1.46x faster than FP32, approximately 15GB memory for 128³ subboxes
- FP32: Higher precision, approximately 30GB memory for 128³ subboxes
- Optimal subbox size: 128×256×256 for 40GB GPU (A100)
- Processing time: Approximately 27 seconds for 512³ volume on A100 (FP16)

## Citation

If you use this code in your research, please cite:

    @software{jamieson2025jax_nbody,
      author = {Jamieson, Drew},
      title = {JAX N-body Emulator},
      year = {2025},
      url = {https://github.com/dsjamieson/jax_nbody_emulator}
    }

And the original papers:
- arXiv:2408.07699
- arXiv:2206.04594
- arXiv:2206.04573

## License

GNU General Public License v3.0
