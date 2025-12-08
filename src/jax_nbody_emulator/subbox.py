# src/jax_nbody_emulator/subbox.py
"""
Subbox processing utilities for handling large volumes.

Splits large 3D volumes into smaller overlapping subboxes for GPU processing,
then reassembles the results with proper boundary handling.
"""

from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm

from .cosmology import D, vel_norm


@dataclass
class SubboxConfig:
    """Configuration for subbox processing."""
    in_chan: int
    size: tuple[int, int, int]
    ndiv: tuple[int, int, int]
    padding: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    
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
    Process large boxes through subbox decomposition.
    
    For models that return only displacement field (no velocity).
    """
    
    def __init__(
        self,
        model,
        params,
        config: SubboxConfig,
        dtype: jnp.dtype = jnp.float32,
        batch_size: int = 1
    ):
        """
        Initialize subbox processor.
        
        Args:
            model: Flax model (NBodyEmulator)
            params: Model parameters (already on GPU)
            config: Subbox configuration
            dtype: Data type for computation (float32 or float16)
            batch_size: Number of subboxes to process together (usually 1)
        """
        self.model = model
        self.params = params
        self.config = config
        self.dtype = dtype
        self.batch_size = batch_size
        
        # JIT compile
        self.apply_fn = jax.jit(
            lambda p, x, Om, Dz: model.apply(p, x, Om, Dz)
        )
    
    def process_box(
        self,
        input_box: np.ndarray,
        z: float,
        Om: float,
        desc: str = "Processing subboxes",
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Process entire box through subboxes.
        
        Args:
            input_box: Input displacement field on CPU (C, D, H, W)
            z: Redshift
            Om: Omega_matter
            desc: Progress bar description
            show_progress: Whether to show progress bar
            
        Returns:
            displacement: (C, D, H, W) on CPU as float32
        """
        # Allocate output array on CPU
        dis_out = np.zeros((self.config.in_chan,) + self.config.size, dtype=np.float32)
        
        # Compute cosmology once in FP32
        Dz = D(jnp.array([z]), jnp.array([Om]))[0]
        
        # Convert to working dtype and transfer to GPU
        Dz_jax = jax.device_put(jnp.array([Dz], dtype=self.dtype))
        Om_jax = jax.device_put(jnp.array([Om], dtype=self.dtype))
        
        # Progress bar
        iterator = range(self.config.n_subboxes)
        if show_progress:
            iterator = tqdm(
                iterator,
                desc=desc,
                ncols=80,
                bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            )
        
        # Pipeline: queue all GPU work
        pending_results = []
        
        for idx in iterator:
            # Extract crop
            crop_inds = self.config.all_crop_inds[idx]
            input_crop = input_box[crop_inds]
            
            # Transfer to GPU
            input_crop_gpu = jax.device_put(
                jnp.array(input_crop, dtype=self.dtype)[None]
            )
            
            # Process (async)
            dis_crop = self.apply_fn(self.params, input_crop_gpu, Om_jax, Dz_jax)
            
            # Queue result
            pending_results.append((idx, dis_crop))
        
        # Copy results
        if show_progress:
            pending_results = tqdm(
                pending_results,
                desc=f"{desc} (copying)",
                ncols=80,
                bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt}'
            )
        
        for idx, dis_crop in pending_results:
            dis_crop_cpu = np.array(dis_crop[0]).astype(np.float32)
            add_inds = self.config.all_add_inds[idx]
            dis_out[add_inds] = dis_crop_cpu
        
        return dis_out

class SubboxProcessorVel:
    """
    Process large boxes through subbox decomposition.
    
    For models that return both displacement and velocity fields.
    """
    
    def __init__(
        self,
        model,
        params,
        config: SubboxConfig,
        dtype: jnp.dtype = jnp.float32,
        batch_size: int = 1
    ):
        """
        Initialize subbox processor.
        
        Args:
            model: Flax model (NBodyEmulatorVel)
            params: Model parameters (already on GPU)
            config: Subbox configuration
            dtype: Data type for computation (float32 or float16)
            batch_size: Number of subboxes to process together (usually 1)
        """
        self.model = model
        self.params = params
        self.config = config
        self.dtype = dtype
        self.batch_size = batch_size
        
        # JIT compile
        self.apply_fn = jax.jit(
            lambda p, x, Om, Dz, vel_fac: model.apply(p, x, Om, Dz, vel_fac)
        )
    
    def process_box(
        self,
        input_box: np.ndarray,
        z: float,
        Om: float,
        desc: str = "Processing subboxes",
        show_progress: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Process entire box through subboxes.
        
        Args:
            input_box: Input displacement field on CPU (C, D, H, W)
            z: Redshift
            Om: Omega_matter
            desc: Progress bar description
            show_progress: Whether to show progress bar
            
        Returns:
            (displacement, velocity): Both (C, D, H, W) on CPU as float32
        """
        # Allocate output arrays on CPU
        dis_out = np.zeros((self.config.in_chan,) + self.config.size, dtype=np.float32)
        vel_out = np.zeros((self.config.in_chan,) + self.config.size, dtype=np.float32)
        
        # Compute cosmology once in FP32 (for accuracy)
        Dz = D(jnp.array([z]), jnp.array([Om]))[0]
        vel_fac = vel_norm(jnp.array([z]), jnp.array([Om]))[0]
        
        # Convert to working dtype and transfer to GPU
        Dz_jax = jax.device_put(jnp.array([Dz], dtype=self.dtype))
        vel_fac_jax = jax.device_put(jnp.array([vel_fac], dtype=self.dtype))
        Om_jax = jax.device_put(jnp.array([Om], dtype=self.dtype))
        
        # Progress bar for processing
        iterator = range(self.config.n_subboxes)
        if show_progress:
            iterator = tqdm(
                iterator,
                desc=desc,
                ncols=80,
                bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            )
        
        # Pipeline: queue all GPU work (async dispatch)
        pending_results = []
        
        for idx in iterator:
            # Extract crop on CPU
            crop_inds = self.config.all_crop_inds[idx]
            input_crop = input_box[crop_inds]
            
            # Transfer to GPU and add batch dimension
            input_crop_gpu = jax.device_put(
                jnp.array(input_crop, dtype=self.dtype)[None]
            )
            
            # Process on GPU (async dispatch)
            dis_crop, vel_crop = self.apply_fn(
                self.params, input_crop_gpu, Om_jax, Dz_jax, vel_fac_jax
            )
            
            # Store GPU results (don't transfer yet)
            pending_results.append((idx, dis_crop, vel_crop))
        
        # Progress bar for copying results
        if show_progress:
            pending_results = tqdm(
                pending_results,
                desc=f"{desc} (copying)",
                ncols=80,
                bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt}'
            )
        
        # Transfer all results back to CPU
        for idx, dis_crop, vel_crop in pending_results:
            # Transfer to CPU and convert to FP32
            dis_crop_cpu = np.array(dis_crop[0]).astype(np.float32)
            vel_crop_cpu = np.array(vel_crop[0]).astype(np.float32)
            
            # Add to output (no padding in add_inds)
            add_inds = self.config.all_add_inds[idx]
            dis_out[add_inds] = dis_crop_cpu
            vel_out[add_inds] = vel_crop_cpu
        
        return dis_out, vel_out
