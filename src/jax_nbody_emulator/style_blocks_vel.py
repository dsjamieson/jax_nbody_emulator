"""
High-level blocks composed of styled layers for building neural networks.

This module contains composite blocks that combine multiple styled layers
to create more complex network components like residual blocks and 
resampling blocks.

Copyright (C) 2025 Drew Jamieson
Licensed under GNU GPL v3.0 - see LICENSE file for details.

Author: Drew Jamieson <drew.s.jamieson@gmail.com>
"""

import jax
import jax.numpy as jnp
import flax.linen as nn

from .style_layers_vel import (
    StyleConv3DVel,
    StyleSkip3DVel,
    StyleUpSample3DVel,
    StyleDownSample3DVel,
    LeakyReLUStyledVel,
)

class StyleResampleBlock3DVel(nn.Module):
    """Sequential block for 3D resampling operations."""
    seq: str
    style_size: int
    in_chan: int
    out_chan: int
    eps: float = 1e-8
    dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(self, x, s, dx=None):
        """Forward pass through resample block."""
        mid_chan = max(self.in_chan, self.out_chan)
        conv_idx = 0  # All conv layers (U/D) share same counter
        act_idx = 0
        
        num_conv = self.seq.count('U') + self.seq.count('D')
        
        for layer_type in self.seq:
            if layer_type == 'U':
                current_in_chan = self.in_chan if conv_idx == 0 else mid_chan
                current_out_chan = self.out_chan if conv_idx == num_conv - 1 else mid_chan
                
                layer = StyleUpSample3DVel(
                    in_chan=current_in_chan,
                    out_chan=current_out_chan,
                    style_size=self.style_size,
                    eps=self.eps,
                    name=f'conv_{conv_idx}',
                    dtype=self.dtype
                )
                x, dx = layer(x, s, dx)
                conv_idx += 1
                
            elif layer_type == 'D':
                current_in_chan = self.in_chan if conv_idx == 0 else mid_chan
                current_out_chan = self.out_chan if conv_idx == num_conv - 1 else mid_chan
                
                layer = StyleDownSample3DVel(
                    in_chan=current_in_chan,
                    out_chan=current_out_chan,
                    style_size=self.style_size,
                    eps=self.eps,
                    name=f'conv_{conv_idx}',
                    dtype=self.dtype
                )
                x, dx = layer(x, s, dx)
                conv_idx += 1
                
            elif layer_type == 'A':
                layer = LeakyReLUStyledVel(name=f'act_{act_idx}', dtype=self.dtype)
                x, dx = layer(x, s, dx)
                act_idx += 1
                
            else:
                raise ValueError(f'Layer type "{layer_type}" not supported.')
        
        return x, dx

class StyleResNetBlock3DVel(nn.Module):
    """3D ResNet-style block with style conditioning."""
    seq: str
    style_size: int
    in_chan: int
    out_chan: int
    eps: float = 1e-8
    dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(self, x, s, dx=None):
        """Forward pass through ResNet block."""
        # Check if final activation is needed
        if self.seq[-1] == 'A':
            main_seq = self.seq[:-1]
            last_act = True
        else:
            main_seq = self.seq
            last_act = False
        
        # Skip connection
        skip = StyleSkip3DVel(
            in_chan=self.in_chan,
            out_chan=self.out_chan,
            style_size=self.style_size,
            eps=self.eps,
            name='skip',
            dtype=self.dtype
        )
        y, dy = skip(x, s, dx)
        
        # Count convolutions for cropping
        num_conv = main_seq.count('C')
        
        # Crop skip connection
        if num_conv > 0:
            crop = num_conv
            y = y[:, :, crop:-crop, crop:-crop, crop:-crop]
            dy = dy[:, :, crop:-crop, crop:-crop, crop:-crop]
        
        # Main path - track conv index separately
        mid_chan = max(self.in_chan, self.out_chan)
        conv_idx = 0  # Separate counter for conv layers only
        act_idx = 0   # Separate counter for activation layers
        
        for layer_type in main_seq:
            # Determine channels for this layer
            if layer_type == 'C':
                current_in_chan = self.in_chan if conv_idx == 0 else mid_chan
                current_out_chan = self.out_chan if conv_idx == num_conv - 1 else mid_chan
                
                layer = StyleConv3DVel(
                    in_chan=current_in_chan,
                    out_chan=current_out_chan,
                    style_size=self.style_size,
                    eps=self.eps,
                    name=f'conv_{conv_idx}',
                    dtype=self.dtype
                )
                x, dx = layer(x, s, dx)
                conv_idx += 1
                
            elif layer_type == 'A':
                layer = LeakyReLUStyledVel(name=f'act_{act_idx}', dtype=self.dtype)
                x, dx = layer(x, s, dx)
                act_idx += 1
                
            else:
                raise ValueError(
                    f'Layer type "{layer_type}" not supported. '
                    f'Use C (conv) or A (activation).'
                )
        
        # Residual addition
        x = x + y
        dx = dx + dy
        
        # Optional final activation
        if last_act:
            act = LeakyReLUStyledVel(name='final_act', dtype=self.dtype)
            x, dx = act(x, s, dx)
        
        return x, dx

