# N-Body Emulator

A JAX-based neural network emulator for N-body cosmological simulations.
Based on the PyTorch version trained with map2map:
https://github.com/eelregit/map2map
https://github.com/dsjamieson/map2map_emu

We kindly request to cite these papers in scientific applications of this code:
arXiv:2408.07699
arXiv:2206.04594
arXiv:2206.04573

## Installation

### From source (editable installation)
```bash
git clone https://github.com/dsjamieson/jax_nbody_emulator.git
cd NbodyEmulator
pip install -e .
```

### Development installation
```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import jax.numpy as jnp
from nbody_emulator import NBodyEmulator, D, H
from nbody_emulator_vel import NBodyEmulatorVel

# Create model without velocity computation
model = NBodyEmulator(
    style_size=2,
    in_chan=3,
    out_chan=3,
)


# Create model with velocity computation
model_vel = NBodyEmulatorVel(
    style_size=2,
    in_chan=3,
    out_chan=3,
)

# Example usage
key = jax.random.PRNGKey(0)
params = model.init(key, 
                   jnp.ones((1, 1, 128, 128, 128)),
                   jnp.array([0.3]), 
                   jnp.array([1.0]))

# Compute cosmological quantities
omega_m = 0.3
redshift = 0.5
growth_factor = D(redshift, omega_m)
hubble_param = H(redshift, omega_m)


```

## Features

- JAX-accelerated cosmological calculations
- Explicit forward-mode derivatives for velocities
- StyleGAN-inspired 3D U-Net architecture
- Modular design for easy extension
- Full type hints and documentation

## License

GNU GENERAL PUBLIC LICENSE
