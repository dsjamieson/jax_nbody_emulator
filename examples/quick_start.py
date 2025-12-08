import jax.numpy as jnp
from nbody_emulator import NBodyEmulator, D, H

# Create model
model = NBodyEmulator(
    style_size=2,
    in_chan=1,
    out_chan=1,
    spatial_in_shape=(1, 1, 128, 128, 128)
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
