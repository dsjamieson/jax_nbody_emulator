"""
Main N-body emulator model implementation.

This module contains the primary NBodyEmulator class that implements
a 3D U-Net-like architecture with style conditioning for cosmological
N-body simulations.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn

from .style_blocks import StyleResNetBlock3D, StyleResampleBlock3D

class NBodyEmulator(nn.Module):
    """
    3D U-Net-like neural network with style conditioning for N-body simulations.
    
    Processes 3D volumetric data through encoder-bottleneck-decoder architecture
    with skip connections. Network behavior is conditioned by cosmological
    parameters (Om and Dz).
    
    Attributes:
        style_size: Dimensionality of style vector (default: 2 for Om, Dz)
        in_chan: Number of input channels
        out_chan: Number of output channels
        mid_chan: Number of channels in intermediate layers
        eps: Small constant for numerical stability
    """
    style_size: int = 2
    in_chan: int = 3
    out_chan: int = 3
    mid_chan: int = 64
    eps: float = 1e-8
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        """Initialize all network layers."""
        mid_chan_1 = self.mid_chan
        mid_chan_2 = 2 * mid_chan_1  # For concatenation with skip connections
        
        # Encoder Path (Downsampling)
        self.conv_l00 = StyleResNetBlock3D(
            'CACA', self.style_size, self.in_chan, mid_chan_1, eps=self.eps, dtype=self.dtype
        )
        self.conv_l01 = StyleResNetBlock3D(
            'CACA', self.style_size, mid_chan_1, mid_chan_1, eps=self.eps, dtype=self.dtype
        )
        self.down_l0 = StyleResampleBlock3D(
            'DA', self.style_size, mid_chan_1, mid_chan_1, eps=self.eps, dtype=self.dtype
        )
        
        self.conv_l1 = StyleResNetBlock3D(
            'CACA', self.style_size, mid_chan_1, mid_chan_1, eps=self.eps, dtype=self.dtype
        )
        self.down_l1 = StyleResampleBlock3D(
            'DA', self.style_size, mid_chan_1, mid_chan_1, eps=self.eps, dtype=self.dtype
        )
        
        self.conv_l2 = StyleResNetBlock3D(
            'CACA', self.style_size, mid_chan_1, mid_chan_1, eps=self.eps, dtype=self.dtype
        )
        self.down_l2 = StyleResampleBlock3D(
            'DA', self.style_size, mid_chan_1, mid_chan_1, eps=self.eps, dtype=self.dtype
        )
        
        # Bottleneck
        self.conv_c = StyleResNetBlock3D(
            'CACA', self.style_size, mid_chan_1, mid_chan_1, eps=self.eps, dtype=self.dtype
        )
        
        # Decoder Path (Upsampling)
        self.up_r2 = StyleResampleBlock3D(
            'UA', self.style_size, mid_chan_1, mid_chan_1, eps=self.eps, dtype=self.dtype
        )
        self.conv_r2 = StyleResNetBlock3D(
            'CACA', self.style_size, mid_chan_2, mid_chan_1, eps=self.eps, dtype=self.dtype
        )
        
        self.up_r1 = StyleResampleBlock3D(
            'UA', self.style_size, mid_chan_1, mid_chan_1, eps=self.eps, dtype=self.dtype
        )
        self.conv_r1 = StyleResNetBlock3D(
            'CACA', self.style_size, mid_chan_2, mid_chan_1, eps=self.eps, dtype=self.dtype
        )
        
        self.up_r0 = StyleResampleBlock3D(
            'UA', self.style_size, mid_chan_1, mid_chan_1, eps=self.eps, dtype=self.dtype
        )
        self.conv_r00 = StyleResNetBlock3D(
            'CACA', self.style_size, mid_chan_2, mid_chan_1, eps=self.eps, dtype=self.dtype
        )
        self.conv_r01 = StyleResNetBlock3D(
            'CAC', self.style_size, mid_chan_1, self.out_chan, eps=self.eps, dtype=self.dtype
        )
    
    def __call__(self, x, Om, Dz):
        """
        Forward pass of the NBodyEmulator.
        
        Args:
            x: Input 3D data tensor of shape (B, C_in, D, H, W)
            Om: Omega_matter cosmological parameter of shape (B,)
            Dz: Linear growth factors at output redshifts of shape (B,)
            
        Returns:
            displacement:
                - displacement: Predicted displacement field (B, C_out, D', H', W')
        """
        x = x.astype(self.dtype)
        Om = Om.astype(self.dtype)
        Dz = Dz.astype(self.dtype)

        # Apply growth factor scaling to input
        # Factor of 6 is from original model input normalization
        x = x * (Dz[:, None, None, None, None] / 6.)

        # Store cropped input for final residual connection
        # Assumes input spatial dimensions are large enough (e.g., 256^3)
        x0 = x[:, :, 48:-48, 48:-48, 48:-48]
        
        # Create style vector from Om and Dz
        s0 = (Om - 0.3) * 5.
        s1 = Dz - 1.
        s = jnp.stack([s0, s1], axis=-1)  # Shape: (B, 2)
        
        # ===== Encoder Path =====
        x = self.conv_l00(x, s)
        y0 = self.conv_l01(x, s)
        
        x = self.down_l0(y0, s)
        # Crop skip connection to match decoder dimensions
        y0 = y0[:, :, 40:-40, 40:-40, 40:-40]
        
        y1 = self.conv_l1(x, s)
        x = self.down_l1(y1, s)
        # Crop skip connection
        y1 = y1[:, :, 16:-16, 16:-16, 16:-16]
        
        y2 = self.conv_l2(x, s)
        x = self.down_l2(y2, s)
        # Crop skip connection
        y2 = y2[:, :, 4:-4, 4:-4, 4:-4]
        
        # ===== Bottleneck =====
        x = self.conv_c(x, s)

        # ===== Decoder Path =====
        x = self.up_r2(x, s)
        # Concatenate with skip connection
        x = jnp.concatenate([y2, x], axis=1)
        x = self.conv_r2(x, s)

        x = self.up_r1(x, s)
        # Concatenate with skip connection
        x = jnp.concatenate([y1, x], axis=1)
        x = self.conv_r1(x, s)

        x = self.up_r0(x, s)
        # Concatenate with skip connection
        x = jnp.concatenate([y0, x], axis=1)
        x = self.conv_r00(x, s)
        x = self.conv_r01(x, s)
        
        # ===== Final Residual Connection =====
        # Displacement field
        displacement = (x + x0) * 6.
        
        return displacement
    
