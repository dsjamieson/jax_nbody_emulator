"""
StyleGAN-inspired 3D convolutional layers with style conditioning.

This module contains the basic building blocks for styled 3D convolutions,
including the base class and specific layer implementations.

Copyright (C) 2025 Drew Jamieson
Licensed under GNU GPL v3.0 - see LICENSE file for details.

Author: Drew Jamieson <drew.s.jamieson@gmail.com>
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import vmap
from functools import partial

class StyleBase3D(nn.Module):
    """
    Base class for standard 3D style-conditioned convolutions.
    
    Handles weight modulation, demodulation, and standard convolution.
    Simpler, JAX-idiomatic implementation without explicit shape tracking.
    """
    in_chan: int
    out_chan: int
    kernel_size: int = 3
    stride: int = 1
    style_size: int = 2
    eps: float = 1e-8
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, s):
        """
        Args:
            x: Input (B, C_in, D, H, W) or (C_in, D, H, W)
            s: Style (B, style_size) or (style_size,)
            
        Returns:
            (y, dy): Output and tangent
        """

        # Add batch dim if needed
        x_unbatched = x.ndim == 4
        if x_unbatched:
            x = x[None]
        if s.ndim == 1:
            s = s[None]
        
        # Style transformation
        style_weight = self.param('style_weight', 
                                   nn.initializers.lecun_normal(),
                                   (self.in_chan, self.style_size),
                                   self.dtype)
        style_bias = self.param('style_bias',
                               nn.initializers.ones,
                               (self.in_chan,),
                               self.dtype)
        
        s_mod = jnp.dot(s, style_weight.T) + style_bias
        
        # Convolution weights
        kernel_shape = (self.kernel_size,) * 3
        weight = self.param('weight',
                           nn.initializers.lecun_normal(),
                           (self.out_chan, self.in_chan, *kernel_shape), 
                           self.dtype)
        bias = self.param('bias',
                         nn.initializers.zeros,
                         (self.out_chan,),
                         self.dtype)
        
        # Weight modulation
        # s_mod: (B, C_in) -> (B, 1, C_in, 1, 1, 1)
        s_mod = s_mod[:, None, :, None, None, None]
        
        # w: (C_out, C_in, K, K, K) -> (B, C_out, C_in, K, K, K)
        w = weight[None] * s_mod
        
        # Demodulation (normalize over spatial + input channels)
        norm = jnp.sqrt(jnp.sum(w**2, axis=(2,3,4,5), keepdims=True) + jnp.array(self.eps, dtype=self.dtype))
        
        w_normalized = w / norm
        
        # Convolution using vmap
        def single_conv(x_i, w_i, b_i):
            out = jax.lax.conv_general_dilated(
                lhs=x_i[None],
                rhs=w_i,
                window_strides=(self.stride,) * 3,
                padding='VALID',
                dimension_numbers=('NCDHW', 'OIDHW', 'NCDHW')
            )
            return out[0] + b_i[:,None,None,None]
        
        
        if x.shape[0] == 1 :
            y = single_conv(x[0], w_normalized[0], bias)[None]
        else :
            # Vectorize over batch
            y = vmap(single_conv, in_axes=(0, 0, None))(x, w_normalized, bias)
        
        # Remove batch if originally unbatched
        if x_unbatched:
            y = y[0]
        
        return y


class StyleConvTransposeBase3D(nn.Module):
    """
    Base class for upsampling 3D style-conditioned convolutions.
    
    Uses lhs_dilation for efficient upsampling. Avoids checkerboard artifacts
    and is more numerically stable than true transposed convolution.
    """
    in_chan: int
    out_chan: int
    kernel_size: int = 2
    stride: int = 1
    style_size: int = 2
    eps: float = 1e-8
    dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(self, x, s):
        """
        Args:
            x: Input (B, C_in, D, H, W) or (C_in, D, H, W)
            s: Style (B, style_size) or (style_size,)
            
        Returns:
            (y, dy): Upsampled output and tangent
        """

        # Add batch dim if needed
        x_unbatched = x.ndim == 4
        if x_unbatched:
            x = x[None]
        if s.ndim == 1:
            s = s[None]
        
        # Style transformation
        style_weight = self.param('style_weight', 
                                   nn.initializers.lecun_normal(),
                                   (self.in_chan, self.style_size),
                                   self.dtype)
        style_bias = self.param('style_bias',
                               nn.initializers.ones,
                               (self.in_chan,),
                               self.dtype)
        
        s_mod = jnp.dot(s, style_weight.T) + style_bias
        
        # Weights
        kernel_shape = (self.kernel_size,) * 3
        weight = self.param('weight',
                           nn.initializers.lecun_normal(),
                           (self.out_chan, self.in_chan, *kernel_shape),
                           self.dtype)
        bias = self.param('bias',
                         nn.initializers.zeros,
                         (self.out_chan,),
                         self.dtype)
        
        # Weight modulation
        s_mod = s_mod[:, None, :, None, None, None]
        
        w = weight[None] * s_mod
        
        # Demodulation
        norm = jnp.sqrt(jnp.sum(w**2, axis=(2,3,4,5), keepdims=True) + jnp.array(self.eps, dtype=self.dtype))
        
        w_normalized = w / norm
        
        def single_upsample_conv(x_i, w_i, b_i):
            out = jax.lax.conv_general_dilated(
                lhs=x_i[None],
                rhs=w_i,
                window_strides=(self.stride,) * 3,
                padding=((1, 1),) * 3,  # Computed here
                lhs_dilation=(2, 2, 2),  # Computed here
                dimension_numbers=('NCDHW', 'OIDHW', 'NCDHW')
            )
            return out[0] + b_i[:,None,None,None]
        
        if x.shape[0] == 1 :
            y = single_upsample_conv(x[0], w_normalized[0], bias)[None]
        else :
            # Vectorize over batch
            y = vmap(single_upsample_conv, in_axes=(0, 0, None))(x, w_normalized, bias)
        
        # Remove batch if originally unbatched
        if x_unbatched:
            y = y[0]
        
        return y
    
class LeakyReLUStyled(nn.Module):
    """Leaky ReLU with tangent computation."""
    negative_slope: float = 0.01
    dtype: jnp.dtype = jnp.float32

    def __call__(self, x, s):
        return jax.nn.leaky_relu(x, negative_slope=jnp.array(self.negative_slope, dtype=self.dtype))

# Specialized layers using partial
StyleConv3D = partial(StyleBase3D, kernel_size=3, stride=1)
StyleSkip3D = partial(StyleBase3D, kernel_size=1, stride=1)
StyleDownSample3D = partial(StyleBase3D, kernel_size=2, stride=2)
StyleUpSample3D = StyleConvTransposeBase3D  # No partial needed, defaults are correct
