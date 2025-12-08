"""
Quick start for the JAX N-body emulator.

Copyright (C) 2025 Drew Jamieson
Licensed under GNU GPL v3.0 - see LICENSE file for details.

Author: Drew Jamieson <drew.s.jamieson@gmail.com>
"""

import jax
import jax.numpy as jnp
import jax_nbody_emulator as jne
from jax_nbody_emulator import NBodyEmulator, NBodyEmulatorVel, D, H, vel_norm

# Run with lower precision for speed
emulator_dtype = jnp.bfloat16

# Load emulator parameters
nbody_emulator_params = jne.load_default_parameters(emulator_dtype)

# Create emulator with velocity output
emulator = NBodyEmulatorVel(dtype = emulator_dtype)

# Example usage
key = jax.random.PRNGKey(0)

# Create a simple test input
pad = 48 # The full model needs 48 voxels of periodic padding on all sides
ngrid_in = 32 # Size of input without padding 
ngrid_pad = ngrid_in + 2 * pad # Size of input with padding
input_shape = (1, 3, ngrid_pad, ngrid_pad, ngrid_pad)  # The first element is the batch dimension
dis_in = jax.random.normal(key, input_shape) # Input represents z=0 displacement field

# Compute cosmological quantities
omega_m = jnp.array([0.3])
redshift = jnp.array([0.5])
growth_factor = jne.D(redshift, omega_m)
vel_factor = jne.vel_norm(redshift, omega_m)

# Run the emulator
dis_out, vel_out = emulator.apply(nbody_emulator_params, dis_in, omega_m, growth_factor, vel_factor)
