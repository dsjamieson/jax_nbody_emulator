"""
Subbox processing utilities for handling large volumes.

Splits large 3D volumes into smaller overlapping subboxes for GPU processing,
then reassembles the results with proper boundary handling.

Copyright (C) 2025 Drew Jamieson
Licensed under GNU GPL v3.0 - see LICENSE file for details.

Author: Drew Jamieson <drew.s.jamieson@gmail.com>
"""

from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm

from .cosmology import growth_factor, vel_norm
from .nbody_emulator_core import NBodyEmulatorCore
from .nbody_emulator_vel_core import NBodyEmulatorVelCore
from .style_nbody_emulator_core import StyleNBodyEmulatorCore
from .style_nbody_emulator_vel_core import StyleNBodyEmulatorVelCore

@dataclass
class SubboxConfig:
    """
    Configuration for subbox processing.
    
    Attributes:
        size: Full box size (D, H, W)
        ndiv: Number of divisions along each dimension
        dtype: Precision for GPU computation (jnp.float16 or jnp.float32)
        output_dtype: Precision for output arrays (np.float16 or np.float32)
        in_chan: Number of input channels (default: 3 for displacement)
        padding: Padding on each side for each dimension
    """
    size: tuple[int, int, int]
    ndiv: tuple[int, int, int]
    dtype: jnp.dtype = jnp.float32
    output_dtype: np.dtype = np.float32
    in_chan: int = 3
    padding: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] = ((48, 48), (48, 48), (48, 48))

    def __post_init__(self):
        """Precompute all crop indices on CPU."""
        self.NDIM = 3
        self.n_subboxes = np.prod(self.ndiv)
        self.crop_size = tuple(s // d for s, d in zip(self.size, self.ndiv))
        
        # Precompute all crop and add indices
        self.all_crop_inds = []
        self.all_add_inds = []
        
        for idx in range(self.n_subboxes):
            crop_inds, add_inds = self._compute_indices(idx)
            self.all_crop_inds.append(crop_inds)
            self.all_add_inds.append(add_inds)
    
    def _get_anchor(self, idx: int) -> tuple[int, int, int]:
        """Compute anchor point for subbox index."""
        return (
            (idx // (self.ndiv[1] * self.ndiv[2])) * self.crop_size[0],
            ((idx // self.ndiv[2]) % self.ndiv[1]) * self.crop_size[1],
            (idx % self.ndiv[2]) * self.crop_size[2]
        )
    
    def _compute_indices(self, idx: int) -> tuple[tuple, tuple]:
        """Compute crop and add indices for a subbox."""
        anchor = self._get_anchor(idx)
        
        # Crop indices (with padding)
        crop_inds = self._get_crop_inds(anchor, self.crop_size, self.padding)
        
        # Add indices (no padding)
        no_pad = ((0, 0),) * self.NDIM
        add_inds = self._get_crop_inds(anchor, self.crop_size, no_pad)
        
        return crop_inds, add_inds
    
    def _get_crop_inds(
        self, 
        anchor: tuple[int, int, int],
        crop: tuple[int, int, int],
        pad: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    ) -> tuple:
        """Get indices for cropping with periodic boundary conditions."""
        ind = [slice(None)]  # Channel dimension
        
        for d, (a, c, (p0, p1), s) in enumerate(zip(anchor, crop, pad, self.size)):
            start = a - p0
            end = a + c + p1
            i = np.arange(start, end) % s  # Periodic BC
            # Reshape for broadcasting
            ind.append(i.reshape((-1,) + (1,) * (self.NDIM - d - 1)))
        
        return tuple(ind)

class SubboxProcessor:
    """
    Unified subbox processor for all model variants.
    
    Handles models with/without premodulation (pre/on-the-fly Style) and with/without velocity output.
    """
    
    def __init__(
        self,
        model,
        params,
        config: SubboxConfig,
    ):
        """
        Initialize subbox processor.
        
        Args:
            model: Flax model (NBodyEmulator variant)
            params: Model parameters (already on GPU)
            config: Subbox configuration
        """
        self.model = model
        self.params = params
        self.config = config

        model_type = type(model)

        if model_type == NBodyEmulatorCore or model_type == NBodyEmulatorVelCore:
            self.premodulate = True
        elif model_type == StyleNBodyEmulatorCore or model_type == StyleNBodyEmulatorVelCore:
            self.premodulate = False

        if model_type == NBodyEmulatorVelCore or model_type == StyleNBodyEmulatorVelCore:
            self.compute_vel = True
        elif model_type == NBodyEmulatorCore or model_type == StyleNBodyEmulatorCore:
            self.compute_vel = False
        
        # JIT compile the appropriate apply function
        self.apply_fn = jax.jit(model.apply)
    
    def process_box(
        self,
        input_box: np.ndarray,
        z: float,
        Om: float,
        desc: str = "Processing subboxes",
        show_progress: bool = True
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Process entire box through subboxes.
        
        Args:
            input_box: Input displacement field on CPU (C, D, H, W)
            z: Redshift
            Om: Omega_matter
            desc: Progress bar description
            show_progress: Whether to show progress bar
            
        Returns:
            If compute_vel=False: displacement (C, D, H, W) on CPU
            If compute_vel=True: (displacement, velocity) tuple
            
        Note:
            Output dtype is determined by config.output_dtype (default: np.float32).
            Use np.float16 for large volumes to reduce memory usage.
        """
        output_dtype = self.config.output_dtype
        
        # Allocate output arrays on CPU
        dis_out = np.zeros((self.config.in_chan,) + self.config.size, dtype=output_dtype)
        if self.compute_vel:
            vel_out = np.zeros((self.config.in_chan,) + self.config.size, dtype=output_dtype)
        
        # Compute cosmology once in FP32 (for accuracy)
        Dz = jnp.atleast_1d(growth_factor(z, Om))
        
        # Prepare additional inputs based on mode
        vel_fac = None
        if self.compute_vel:
            vel_fac = jnp.atleast_1d(vel_norm(z, Om))
        
        if not self.premodulate:
            Om = jnp.atleast_1d(Om)
        else:
            Om = None
        
        # Progress bar
        iterator = range(self.config.n_subboxes)
        if show_progress:
            iterator = tqdm(
                iterator,
                desc=desc,
                ncols=80,
                bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            )
        
        for idx in iterator:
            # Extract crop on CPU
            crop_inds = self.config.all_crop_inds[idx]
            
            # Transfer to GPU and add batch dimension
            input_crop_gpu = jax.device_put(
                jnp.array(input_box[crop_inds], dtype=self.config.dtype)[None]
            )
            
            # Call model with appropriate signature
            result = self._apply_model(input_crop_gpu, Om, Dz, vel_fac)
            
            # Transfer to CPU immediately and convert to output dtype
            add_inds = self.config.all_add_inds[idx]
            
            if self.compute_vel:
                dis_crop, vel_crop = result
                dis_out[add_inds] = np.asarray(dis_crop[0]).astype(output_dtype)
                vel_out[add_inds] = np.asarray(vel_crop[0]).astype(output_dtype)
            else:
                dis_out[add_inds] = np.asarray(result[0]).astype(output_dtype)
        
        if self.compute_vel:
            return dis_out, vel_out
        return dis_out
    
    def _apply_model(self, x, Om, Dz, vel_fac):
        """Dispatch to model with correct signature."""
        x = x.astype(self.config.dtype)
        if self.premodulate:
            if self.compute_vel:
                return self.apply_fn(self.params, x, Dz, vel_fac)
            else:
                return self.apply_fn(self.params, x, Dz)
        else:
            if self.compute_vel:
                return self.apply_fn(self.params, x, Om, Dz, vel_fac)
            else:
                return self.apply_fn(self.params, x, Om, Dz)
