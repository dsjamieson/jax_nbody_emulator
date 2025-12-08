"""
N-Body Emulator: A JAX-based neural network emulator for cosmological simulations.

Copyright (C) 2025 Drew Jamieson
Licensed under GNU GPL v3.0 - see LICENSE file for details.

Author: Drew Jamieson <drew.s.jamieson@gmail.com>
"""

from importlib import resources
import numpy as np
import jax.numpy as jnp
import jax


def load_default_parameters(dtype=jnp.float32):
    """
    Load the default pretrained N-body emulator parameters.
    
    Parameters
    ----------
    dtype : jnp.dtype, optional
        Desired dtype for all parameter arrays (default: jnp.float32).
        Common options: jnp.float32, jnp.float16, jnp.bfloat16
    
    Returns
    -------
    dict
        Dictionary containing the parameter pytree and any metadata.
        Access parameters with data['params'].
    
    Examples
    --------
    >>> # Load in FP32 (default)
    >>> data = load_default_parameters()
    >>> params = data['params']
    
    >>> # Load in FP16 for faster inference
    >>> data = load_default_parameters(dtype=jnp.float16)
    >>> params_fp16 = data['params']
    """
    with resources.files(__package__).joinpath(
        "model_parameters/nbody_emulator_params.npz"
    ).open("rb") as f:
        # Load as NumPy arrays first
        raw = np.load(f, allow_pickle=True)
        data = {k: raw[k] for k in raw.files}
    
    # Extract and convert params
    params = data.get("params")
    if params is not None:
        params = params.item()  # Unwrap the pytree from numpy object array
        
        # Convert to JAX arrays with desired dtype
        def convert_array(x):
            if isinstance(x, (np.ndarray, jnp.ndarray)):
                # Handle both NumPy and JAX arrays
                # Convert to target dtype, preserving JAX array status
                if isinstance(x, np.ndarray):
                    return jnp.asarray(x, dtype=dtype)
                else:
                    return x.astype(dtype)
            return x
        
        # Use tree_map with is_leaf to ensure we traverse everything
        params = jax.tree_util.tree_map(
            convert_array, 
            params,
            is_leaf=lambda x: isinstance(x, (np.ndarray, jnp.ndarray))
        )
        data["params"] = params
    
    return data



from .cosmology import D, H, f, dlogH_dloga, vel_norm, acc_norm
from .style_layers import (
    StyleBase3D,
    StyleConv3D,
    StyleSkip3D,
    StyleDownSample3D,
    StyleUpSample3D,
    LeakyReLUStyled,
)
from .style_blocks import (
    StyleResampleBlock3D,
    StyleResNetBlock3D,
)
from .nbody_emulator import NBodyEmulator
from .style_layers_vel import (
    StyleBase3DVel,
    StyleConv3DVel,
    StyleSkip3DVel,
    StyleDownSample3DVel,
    StyleUpSample3DVel,
    LeakyReLUStyledVel,
)
from .style_blocks_vel import (
    StyleResampleBlock3DVel,
    StyleResNetBlock3DVel,
)
from .nbody_emulator_vel import NBodyEmulatorVel
from .subbox import SubboxConfig, SubboxProcessor, SubboxProcessorVel


__version__ = "0.1.0"
__author__ = "Drew Jamieson"
__email__ = "drew.s.jamieson@gmail.com"

__all__ = [
    # Cosmology functions
    "D",
    "H", 
    "f",
    "dlogH_dloga",
    "growth_acc",
    "vel_norm",
    "acc_norm",
    # Style layers
    "StyleBase3D",
    "StyleConv3D",
    "StyleSkip3D", 
    "StyleDownSample3D",
    "StyleUpSample3D",
    "LeakyReLUStyled",
    # Style blocks
    "StyleResampleBlock3D",
    "StyleResNetBlock3D",
    # Main model
    "NBodyEmulator",
    # Style layers with velocity
    "StyleBase3DVel",
    "StyleConv3DVel",
    "StyleSkip3DVel", 
    "StyleDownSample3DVel",
    "StyleUpSample3DVel",
    "LeakyReLUStyledVel",
    # Style blocks
    "StyleResampleBlock3DVel",
    "StyleResNetBlock3DVel",
    # Main model
    "NBodyEmulatorVel",
    # Subbox processors
    "SubboxConfig",
    "SubboxProcessor",
    "SubboxProcessorVel"
]
