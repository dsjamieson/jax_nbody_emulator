"""
Factory functions for creating N-body emulator models and processors.

Provides a unified interface for constructing emulator bundles with
the appropriate model variant, parameters, and subbox processor.

Copyright (C) 2025 Drew Jamieson
Licensed under GNU GPL v3.0 - see LICENSE file for details.

Author: Drew Jamieson <drew.s.jamieson@gmail.com>
"""

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import flax.linen as nn

from .cosmology import growth_factor
from .subbox import SubboxConfig, SubboxProcessor


@dataclass
class NBodyEmulator:
    """
    Container for emulator components with convenient access methods.
    
    Attributes:
        model: The underlying Flax model
        params: Model parameters (None if not loaded)
        processor: SubboxProcessor for large volumes (None if not created)
        premodulate: Whether params were premodulated (True=fixed cosmology, False=runtime cosmology), defaut = False
        compute_vel: Whether model returns velocity field, default=True
    """
    model: nn.Module
    params: dict | None
    processor: SubboxProcessor | None
    premodulate: bool = False
    compute_vel: bool = True
    dtype: jnp.dtype = jnp.float32
    
    def apply(self, x, z, Om):
        """
        Apply model directly to input tensor.
        
        Args:
            x: Input tensor (B, C, D, H, W)
            z: Redshift (scalar or array)
            Om: Omega_matter (scalar or array)
            
        Returns:
            If compute_vel=False: displacement (B, C, D', H', W')
            If compute_vel=True: (displacement, velocity) tuple
        """
        if self.params is None:
            raise ValueError("No parameters loaded. Use load_params=True in create_emulator.")
        
        from .cosmology import growth_factor, vel_norm

        z = jnp.atleast_1d(z)
        Om = jnp.atleast_1d(Om)
        Dz = growth_factor(z, Om)
        if self.compute_vel:
            vel_fac = vel_norm(z, Om)

        x = x.astype(self.dtype)
        
        if self.premodulate:
            # Premodulated params - non-style models (Om baked in)
            if self.compute_vel:
                return self.model.apply(self.params, x, Dz, vel_fac)
            else:
                return self.model.apply(self.params, x, Dz)
        else:
            # Style models - Om passed at runtime
            if self.compute_vel:
                return self.model.apply(self.params, x, Om, Dz, vel_fac)
            else:
                return self.model.apply(self.params, x, Om, Dz)
    
    def process_box(
        self,
        input_box,
        z: float,
        Om: float,
        desc: str = "Processing subboxes",
        show_progress: bool = True
    ):
        """
        Process large volume through subbox decomposition.
        
        Args:
            input_box: Input displacement field on CPU (C, D, H, W)
            z: Redshift
            Om: Omega_matter
            desc: Progress bar description
            show_progress: Whether to show progress bar
            
        Returns:
            If compute_vel=False: displacement (C, D, H, W) on CPU as float32
            If compute_vel=True: (displacement, velocity) tuple
        """
        if self.processor is None:
            raise ValueError("No processor created. Use create_processor=True in create_emulator.")
        
        return self.processor.process_box(
            input_box, z, Om, desc=desc, show_progress=show_progress
        )
    
    def __call__(self, x, z, Om):
        """Alias for apply()."""
        return self.apply(x, z, Om)


def load_default_parameters() -> dict:
    """
    Load default pretrained model parameters.
    
    Returns:
        Parameter dictionary
    """
    import numpy as np
    
    params_path = Path(__file__).parent / "model_parameters" / "nbody_emulator_params.npz"
    
    with np.load(params_path, allow_pickle=True) as f:
        params = f['params'].item()
    
    return {'params': params}

def _modulate_weights(style_weight, style_bias, weight, s, eps = 1.e-8) :

        if s.ndim == 1:
            s = s[None]

        s_mod = jnp.dot(s, style_weight.T) + style_bias
        # s_mod: (B, C_in) -> (B, 1, C_in, 1, 1, 1)
        s_mod = s_mod[:, None, :, None, None, None]
        
        # w: (C_out, C_in, K, K, K) -> (B, C_out, C_in, K, K, K)
        w = weight[None] * s_mod
        
        # Demodulation (normalize over spatial + input channels)
        norm = jnp.sqrt(jnp.sum(w**2, axis=(2,3,4,5), keepdims=True) + jnp.array(eps))
        
        w_normalized = w / norm
        
        return w_normalized

def modulate_emulator_parameters(params, z, Om, eps = 1.e-8):
    """
    Preprocess all network parameters for fixed (z, Om).
    
    Returns new params dict with modulated weights.
    """
    
    Dz = growth_factor(z, Om)
    
    # Compute style vector
    s0 = (Om - 0.3) * 5.
    s1 = Dz - 1.
    s = jnp.stack([s0, s1], axis=-1)
    
    # Process each block
    processed_params = {'params':{}}
    for block_name, block_params in params['params'].items():
        processed_params['params'][block_name] = {}
        for layer_name, layer_params in block_params.items():
            if 'style_weight' in layer_params:
                # This layer has style modulation - preprocess it
                w_norm = _modulate_weights(
                    layer_params['style_weight'],
                    layer_params['style_bias'],
                    layer_params['weight'],
                    s,
                    eps=eps
                )
                processed_params['params'][block_name][layer_name] = {
                    'weight': w_norm[0],
                    'bias': layer_params['bias']
                }
            else:
                # Pass through unmodified
                print(f'skipping {block_name} {layer_name}')
                processed_params['params'][block_name][layer_name] = layer_params
    
    return processed_params

def _modulate_weights_vel(style_weight, style_bias, weight, s, dx=None, eps = 1.e-8) :

        if s.ndim == 1:
            s = s[None]

        s_mod = jnp.dot(s, style_weight.T) + style_bias
        # s_mod: (B, C_in) -> (B, 1, C_in, 1, 1, 1)
        s_mod = s_mod[:, None, :, None, None, None]
        ds = jnp.zeros_like(s).at[:, 1].set(1.0)
        ds_mod = jnp.dot(ds, style_weight.T)
        ds_mod = ds_mod[:, None, :, None, None, None]
        
        # w: (C_out, C_in, K, K, K) -> (B, C_out, C_in, K, K, K)
        w = weight[None] * s_mod
        dw_style = weight[None] * ds_mod
        
        # Demodulation (normalize over spatial + input channels)
        norm = jnp.sqrt(jnp.sum(w**2, axis=(2,3,4,5), keepdims=True) + eps)
        dnorm = -jnp.sum(w * dw_style, axis=(2,3,4,5), keepdims=True) / (norm**3)
        
        w_normalized = w / norm
        dw_normalized = (dw_style / norm) + (w * dnorm)

        if dx is None:
            # First layer handling
            Dz = (s[:, 1] + 1.0)[:, None, None, None, None, None]
            dw_normalized = dw_normalized + w_normalized / Dz
        else:
            dw_normalized = dw_normalized
        
        return w_normalized, dw_normalized
    
def modulate_emulator_parameters_vel(params, z, Om, eps = 1.e-8):
    """
    Preprocess all network parameters for fixed (z, Om).
    
    Returns new params dict with modulated weights.
    """

    Dz = growth_factor(z, Om)
    
    # Compute style vector
    s0 = (Om - 0.3) * 5.
    s1 = Dz - 1.
    s = jnp.stack([s0, s1], axis=-1)
    
    # Process each block
    processed_params = {'params':{}}
    # Process each block
    for block_name, block_params in params['params'].items():
        processed_params['params'][block_name] = {}
        for layer_name, layer_params in block_params.items():
            if 'style_weight' in layer_params:
                # Only the first block's conv_0 and skip layers have input that is linear in Dz
                if block_name == 'conv_l00' and (layer_name == 'conv_0' or layer_name == 'skip') :
                    dx = None
                else :
                    dx = 1
                # This layer has style modulation - preprocess it
                w_norm, dw_norm = _modulate_weights_vel(
                    layer_params['style_weight'],
                    layer_params['style_bias'],
                    layer_params['weight'],
                    s,
                    dx=dx,
                    eps=eps
                )
                processed_params['params'][block_name][layer_name] = {
                    'weight': w_norm[0],
                    'dweight': dw_norm[0],
                    'bias': layer_params['bias']
                }
            else:
                # Pass through unmodified
                print(f'skipping {block_name} {layer_name}')
                processed_params['params'][block_name][layer_name] = layer_params

    return processed_params

def create_emulator(
    premodulate: bool = False,
    compute_vel: bool = True,
    load_params: bool = True,
    processor_config: SubboxConfig | None = None,
    premodulate_z: float | None = None,
    premodulate_Om: float | None = None,
    dtype: jnp.dtype | None = None,
    **model_kwargs
) -> NBodyEmulator:
    """
    Factory function to create emulator, optionally with params and processor.
    
    Args:
        premodulate: If True, premodulate params at creation time (fixed cosmology).
                     If False, use Style models (Om passed at runtime).
        compute_vel: If True, model returns both displacement and velocity.
        load_params: If True, load default pretrained parameters.
        processor_config: SubboxConfig for processor (creates processor only if not None).
        premodulate_z: If premodulate=True and load_params=True, premodulate 
                              params at this redshift (required with premodulate_Om).
        premodulate_Om: If premodulate=True and load_params=True, premodulate
                               params at this Omega_matter.
        dtype: dtype of convolution (jnp.float32 or jnp.float16), overwritten by
                    processor_config.dtype if present, otherwise defaults to jnp.float32
        **model_kwargs: Passed to model constructor (in_chan, out_chan, mid_chan, eps).
    
    Returns:
        NBodyEmulator bundle with model, params, and processor.
        
    Raises:
        ValueError: If required arguments are missing.
        
    Examples:
        # Style model with velocity (Om at runtime)
        emulator = create_emulator(
            premodulate=False,
            compute_vel=True,
            load_params=True,
            processor_config=config,
        )
        disp, vel = emulator.process_box(input_box, z=0.0, Om=0.3)
        
        # Fixed-cosmology model (Om and z baked into params)
        emulator = create_emulator(
            premodulate=True,
            compute_vel=False,
            load_params=True,
            processor_config=config,
            premodulate_z=0.0,
            premodulate_Om=0.3,
        )
        disp = emulator.process_box(input_box, z=0.0, Om=0.3)
    """
    
    # Create model
    if premodulate:
        # Premodulate params - use non-style models
        if compute_vel:
            from .nbody_emulator_vel_core import NBodyEmulatorVelCore
            model = NBodyEmulatorVelCore(**model_kwargs)
        else:
            from .nbody_emulator_core import NBodyEmulatorCore
            model = NBodyEmulatorCore(**model_kwargs)
    else:
        # Don't premodulate - use style models (Om at runtime)
        if compute_vel:
            from .style_nbody_emulator_vel_core import StyleNBodyEmulatorVelCore
            model = StyleNBodyEmulatorVelCore(**model_kwargs)
        else:
            from .style_nbody_emulator_core import StyleNBodyEmulatorCore
            model = StyleNBodyEmulatorCore(**model_kwargs)
    
    # Load parameters
    params = None
    if load_params:
        params = load_default_parameters()
        
        # Premodulate params if requested
        if premodulate:
            if premodulate_z is None or premodulate_Om is None:
                raise ValueError(
                    "premodulate_z and premodulate_Om are required "
                    "when premodulate=True and load_params=True"
                )
            if compute_vel:
                params = modulate_emulator_parameters_vel(
                    params, 
                    premodulate_z, 
                    premodulate_Om 
                )
            else:
                params = modulate_emulator_parameters(
                    params, 
                    premodulate_z, 
                    premodulate_Om, 
                )
    
    # Create processor
    processor = None
    if processor_config is not None:
        processor = SubboxProcessor(model, params, processor_config)

    # Set dtype based on processor_config (overwrites) or chosen dtype
    if processor_config is not None :
        dtype = processor_config.dtype
    elif dtype is None :
        dtype = jnp.float32
    
    return NBodyEmulator(
        model=model,
        params=params,
        processor=processor,
        premodulate=premodulate,
        compute_vel=compute_vel,
        dtype=dtype
    )
