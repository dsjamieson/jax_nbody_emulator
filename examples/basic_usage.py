"""
Basic usage example for the JAX N-body emulator.

Quick start (copy-paste this):

    import numpy as np
    from jax_nbody_emulator import create_emulator, SubboxConfig
    
    config = SubboxConfig(size=(512, 512, 512), ndiv=(4, 4, 4))
    emulator = create_emulator(compute_vel=True, processor_config=config)
    
    input_displacement = np.random.randn(3, 512, 512, 512).astype(np.float32)
    displacement, velocity = emulator.process_box(input_displacement, z=0.5, Om=0.3)

Copyright (C) 2025 Drew Jamieson
Licensed under GNU GPL v3.0 - see LICENSE file for details.

Author: Drew Jamieson <drew.s.jamieson@gmail.com>
"""

import jax
import jax.numpy as jnp
import numpy as np

from jax_nbody_emulator import (
    create_emulator,
    SubboxConfig,
    load_default_parameters,
    growth_factor,
    hubble_rate,
    growth_rate,
    vel_norm,
)


def main():
    print("=" * 60)
    print("JAX N-body Emulator - Basic Usage Example")
    print("=" * 60)
    
    # =========================================================================
    # Example 1: Quick start with create_emulator (recommended)
    # =========================================================================
    print("\n--- Example 1: Using create_emulator (recommended) ---\n")
    
    # Configure subbox processing for a 128³ volume
    config = SubboxConfig(
        size=(128, 128, 128),
        ndiv=(2, 2, 2),  # Divide into 8 subboxes
    )
    
    # Create emulator with velocity output
    emulator = create_emulator(
        compute_vel=True,
        processor_config=config,
    )
    
    # Create test input (3, 128, 128, 128) - no batch dimension for process_box
    key = jax.random.PRNGKey(42)
    input_displacement = np.array(jax.random.normal(key, (3, 128, 128, 128)))
    
    # Process the box
    z = 0.5
    Om = 0.3
    displacement, velocity = emulator.process_box(input_displacement, z=z, Om=Om)
    
    print(f"Input shape: {input_displacement.shape}")
    print(f"Output displacement shape: {displacement.shape}")
    print(f"Output velocity shape: {velocity.shape}")
    
    # =========================================================================
    # Example 2: Fixed cosmology mode (faster for repeated inference)
    # =========================================================================
    print("\n--- Example 2: Fixed cosmology (premodulated) ---\n")
    
    emulator_premod = create_emulator(
        premodulate=True,
        premodulate_z=0.5,
        premodulate_Om=0.3,
        compute_vel=True,
        processor_config=config,
    )
    
    # Process multiple boxes with same cosmology
    for i in range(3):
        key = jax.random.PRNGKey(i)
        input_box = np.array(jax.random.normal(key, (3, 128, 128, 128)))
        disp, vel = emulator_premod.process_box(input_box, z=0.5, Om=0.3)
        print(f"Box {i+1}: displacement range [{disp.min():.3f}, {disp.max():.3f}]")
    
    # =========================================================================
    # Example 3: Direct model access (for advanced users)
    # =========================================================================
    print("\n--- Example 3: Direct model access ---\n")
    
    from jax_nbody_emulator import StyleNBodyEmulatorVelCore
    
    # Load pretrained parameters
    params = load_default_parameters()
    print(f"Model has {sum(p.size for p in jax.tree.leaves(params)):,} parameters")
    
    # Create model
    model = StyleNBodyEmulatorVelCore()
    
    # Prepare input (batch, channels, D, H, W)
    # Input size must be output_size + 2 * padding (e.g., 32 + 96 = 128)
    pad = 48
    ngrid_out = 32
    ngrid_in = ngrid_out + 2 * pad
    
    x = jax.random.normal(key, (1, 3, ngrid_in, ngrid_in, ngrid_in))
    
    # Compute cosmological quantities
    z_arr = jnp.array([z])
    Om_arr = jnp.array([Om])
    Dz = growth_factor(z_arr, Om_arr)
    vn = vel_norm(z_arr, Om_arr)
    
    # Run inference
    disp_out, vel_out = model.apply(params, x, Om_arr, Dz, vn)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {disp_out.shape}")
    
    # =========================================================================
    # Example 4: Cosmology functions
    # =========================================================================
    print("\n--- Example 4: Cosmology functions ---\n")
    
    z_vals = jnp.array([0.0, 0.5, 1.0, 2.0])
    Om_vals = jnp.array([0.3, 0.3, 0.3, 0.3])
    
    D = growth_factor(z_vals, Om_vals)
    H = hubble_rate(z_vals, Om_vals)
    f = growth_rate(z_vals, Om_vals)
    
    print(f"{'z':>6} {'D(z)':>10} {'H(z)':>12} {'f(z)':>10}")
    print("-" * 42)
    for i in range(len(z_vals)):
        print(f"{z_vals[i]:>6.1f} {D[i]:>10.4f} {H[i]:>12.2f} {f[i]:>10.4f}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
