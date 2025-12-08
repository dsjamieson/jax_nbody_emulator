"""
Basic usage example for the JAX N-body emulator.

Copyright (C) 2025 Drew Jamieson
Licensed under GNU GPL v3.0 - see LICENSE file for details.

Author: Drew Jamieson <drew.s.jamieson@gmail.com>
"""

import jax
import jax.numpy as jnp
import jax_nbody_emulator as jne
from jax_nbody_emulator import NBodyEmulator, NBodyEmulatorVel, D, H, vel_norm

def main():
    # Initialize random key
    key = jax.random.PRNGKey(42)
    
    # Create a simple test input
    pad = 48 # The full model needs 48 voxels of periodic padding on all sides
    ngrid_in = 32 # Size of input without padding 
    ngrid_pad = ngrid_in + 2 * pad # Size of input with padding
    input_shape = (1, 3, ngrid_pad, ngrid_pad, ngrid_pad)  # The first element is the batch dimension
    dis_in = jax.random.normal(key, input_shape) # Input represents z=0 displacement field
    
    # Cosmological parameters
    Om = jnp.array([0.3]) # Omega_m
    z = jnp.array([0.5])  # desired redshift of output
    growth_factor = D(z, Om)

    # Load emulator parameters
    nbody_emulator_params = jne.load_default_parameters()
    print(f"Model has {sum(p.size for p in jax.tree.leaves(nbody_emulator_params))} parameters")
 
    for k in nbody_emulator_params.keys() :
        print(k)
        for kk in nbody_emulator_params[k].keys() :
            print(f"  {kk}")
            for k2 in nbody_emulator_params[k][kk].keys() :
                print(f"    {k2}", end = "")
                for k3 in nbody_emulator_params[k][kk][k2].keys() :
                    print(f" {k3}", end = "")
                print()


    exit()
   
    # Create emulator without velocity output
    emulator = NBodyEmulator(
        style_size=2,
        in_chan=3,
        out_chan=3,
        mid_chan=64
    )

    # Forward pass
    dis_out = emulator.apply(nbody_emulator_params, dis_in, Om, growth_factor)
    
    print(f"Input shape: {dis_in.shape}")
    print(f"Output shape: {dis_out.shape}")
    
    # conversion factor d/dD(z) dis_out --> pecular velocity
    vel_factor = vel_norm(z, Om)

    # Create model with velocity output
    emulator = NBodyEmulatorVel(
        style_size=2,
        in_chan=3,
        out_chan=3,
        mid_chan=64
    )

    # Forward pass
    dis_out, vel_out = emulator.apply(nbody_emulator_params, dis_in, Om, growth_factor, vel_factor)

    print(f"Output shapes with velocities: {dis_out.shape} {vel_out.shape}")

    # Example cosmological calculations
    D_z = D(z[0], Om[0])
    H_z = H(z[0], Om[0])
    
    print(f"Growth factor D(z={z}) = {D_z:.4f}")
    print(f"Hubble parameter H(z={z}) = {H_z:.2f} h km/s/Mpc")

if __name__ == "__main__":
    main()
