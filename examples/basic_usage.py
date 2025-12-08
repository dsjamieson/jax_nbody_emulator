"""
Basic usage example for the N-body emulator.
"""

import jax
import jax.numpy as jnp
from nbody_emulator import NBodyEmulator, D, H

def main():
    # Initialize random key
    key = jax.random.PRNGKey(42)
    
    # Create a simple test input
    input_shape = (1, 1, 64, 64, 64)  # Smaller for example
    x = jax.random.normal(key, input_shape)
    
    # Cosmological parameters
    omega_m = jnp.array([0.3])
    growth_factor = jnp.array([1.0])
    
    # Create model
    model = NBodyEmulator(
        style_size=2,
        in_chan=1,
        out_chan=1,
        spatial_in_shape=input_shape
    )
    
    # Initialize parameters
    params = model.init(key, x, omega_m, growth_factor)
    
    # Forward pass
    output = model.apply(params, x, omega_m, growth_factor)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model has {sum(p.size for p in jax.tree_leaves(params))} parameters")
    
    # Example cosmological calculations
    z = 0.5
    D_z = D(z, omega_m[0])
    H_z = H(z, omega_m[0])
    
    print(f"Growth factor D(z={z}) = {D_z:.4f}")
    print(f"Hubble parameter H(z={z}) = {H_z:.2f} km/s/Mpc")

if __name__ == "__main__":
    main()
