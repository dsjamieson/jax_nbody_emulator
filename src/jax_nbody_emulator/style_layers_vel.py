"""
StyleGAN-inspired 3D convolutional layers with style conditioning and manual forward-mode AD.

This is the "Vel" (velocity) version that computes both outputs and their derivatives
w.r.t. the style parameter s[1] (growth factor Dz) using manual forward-mode automatic
differentiation. Used for predicting both displacement and velocity fields simultaneously.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import vmap
from functools import partial

class StyleBase3DVel(nn.Module):
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
    def __call__(self, x, s, dx=None):
        """
        Args:
            x: Input (B, C_in, D, H, W) or (C_in, D, H, W)
            s: Style (B, style_size) or (style_size,)
            dx: Optional tangent (same shape as x)
            
        Returns:
            (y, dy): Output and tangent
        """
        # Add batch dim if needed
        x_unbatched = x.ndim == 4
        if x_unbatched:
            x = x[None]
            if dx is not None:
                dx = dx[None]
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
        
        # Style tangent (derivative w.r.t. s[:, 1])
        ds = jnp.zeros_like(s).at[:, 1].set(1.0)
        ds_mod = jnp.dot(ds, style_weight.T)
        
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
        ds_mod = ds_mod[:, None, :, None, None, None]
        
        # w: (C_out, C_in, K, K, K) -> (B, C_out, C_in, K, K, K)
        w = weight[None] * s_mod
        dw_style = weight[None] * ds_mod
        
        # Demodulation (normalize over spatial + input channels)
        norm = jnp.sqrt(jnp.sum(w**2, axis=(2,3,4,5), keepdims=True) + jnp.array(self.eps, dtype=self.dtype))
        dnorm = -jnp.sum(w * dw_style, axis=(2,3,4,5), keepdims=True) / (norm**3)
        
        w_normalized = w / norm
        dw_normalized = (dw_style / norm) + (w * dnorm)
        
        # Extract Dz and compute w/Dz term for first layer
        Dz = s[:, 1] + 1.0
        
        if dx is None:
            # First layer: add w/Dz term
            Dz_broadcast = Dz[:, None, None, None, None, None]
            dw_total = dw_normalized + w_normalized / Dz_broadcast
        else:
            dw_total = dw_normalized
        
        # Convolution using vmap
        def single_conv(x_i, w_i):
            out = jax.lax.conv_general_dilated(
                lhs=x_i[None],
                rhs=w_i,
                window_strides=(self.stride,) * 3,
                padding='VALID',
                dimension_numbers=('NCDHW', 'OIDHW', 'NCDHW')
            )
            return out[0]
        
        def single_conv_b(x_i, w_i, b_i):
            out = jax.lax.conv_general_dilated(
                lhs=x_i[None],
                rhs=w_i,
                window_strides=(self.stride,) * 3,
                padding='VALID',
                dimension_numbers=('NCDHW', 'OIDHW', 'NCDHW')
            )
            return out[0] + b_i[:,None,None,None]
        
        
        if x.shape[0] == 1 :
            y = single_conv_b(x[0], w_normalized[0], bias)[None]
            dy = single_conv(x[0], dw_total[0])[None]
        else :
            # Vectorize over batch
            y = vmap(single_conv_b, in_axes=(0, 0, None))(x, w_normalized, bias)
            dy = vmap(single_conv, in_axes=(0, 0))(x, dw_total)

        if dx is not None:
            if x.shape[0] == 1 :
                dy = dy + single_conv(dx[0], w_normalized[0])[None]
            else :
                dy = dy + vmap(single_conv, in_axes=(0, 0))(dx, w_normalized)
        
        # Remove batch if originally unbatched
        if x_unbatched:
            y, dy = y[0], dy[0]
        
        return y, dy


class StyleTransposeBase3DVel(nn.Module):
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
    def __call__(self, x, s, dx=None):
        """
        Args:
            x: Input (B, C_in, D, H, W) or (C_in, D, H, W)
            s: Style (B, style_size) or (style_size,)
            dx: Optional tangent (same shape as x)
            
        Returns:
            (y, dy): Upsampled output and tangent
        """
        # Add batch dim if needed
        x_unbatched = x.ndim == 4
        if x_unbatched:
            x = x[None]
            if dx is not None:
                dx = dx[None]
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
        
        # Style tangent
        ds = jnp.zeros_like(s).at[:, 1].set(1.0)
        ds_mod = jnp.dot(ds, style_weight.T)
        
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
        ds_mod = ds_mod[:, None, :, None, None, None]
        
        w = weight[None] * s_mod
        dw_style = weight[None] * ds_mod
        
        # Demodulation
        norm = jnp.sqrt(jnp.sum(w**2, axis=(2,3,4,5), keepdims=True) + jnp.array(self.eps, dtype=self.dtype))
        dnorm = -jnp.sum(w * dw_style, axis=(2,3,4,5), keepdims=True) / (norm**3)
        
        w_normalized = w / norm
        dw_normalized = (dw_style / norm) + (w * dnorm)
        
        # First layer handling
        Dz = s[:, 1] + 1.0
        
        if dx is None:
            Dz_broadcast = Dz[:, None, None, None, None, None]
            dw_total = dw_normalized + w_normalized / Dz_broadcast
        else:
            dw_total = dw_normalized
        
        # Upsampling convolution (using lhs_dilation)
        def single_upsample_conv(x_i, w_i):
            out = jax.lax.conv_general_dilated(
                lhs=x_i[None],
                rhs=w_i,
                window_strides=(self.stride,) * 3,
                padding=((1, 1),) * 3,  # Computed here
                lhs_dilation=(2, 2, 2),  # Computed here
                dimension_numbers=('NCDHW', 'OIDHW', 'NCDHW')
            )
            return out[0]
        
        def single_upsample_conv_b(x_i, w_i, b_i):
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
            y = single_upsample_conv_b(x[0], w_normalized[0], bias)[None]
            dy = single_upsample_conv(x[0], dw_total[0])[None]
        else :
            # Vectorize over batch
            y = vmap(single_upsample_conv_b, in_axes=(0, 0, None))(x, w_normalized, bias)
            dy = vmap(single_upsample_conv, in_axes=(0, 0))(x, dw_total)
        
        if dx is not None:
            if x.shape[0] == 1 :
                dy = dy + single_upsample_conv(dx[0], w_normalized[0])[None]
            else :
                dy = dy + vmap(single_upsample_conv, in_axes=(0, 0))(dx, w_normalized)
        
        # Remove batch if originally unbatched
        if x_unbatched:
            y, dy = y[0], dy[0]
        
        return y, dy
    
class LeakyReLUStyledVel(nn.Module):
    """Leaky ReLU with tangent computation."""
    negative_slope: float = 0.01
    dtype: jnp.dtype = jnp.float32
    
    def __call__(self, x, s, dx):
        slope = jnp.array(self.negative_slope, dtype=self.dtype)
        y = jax.nn.leaky_relu(x, negative_slope=slope)
        dy = jnp.where(x > 0, dx, slope * dx)
        return y, dy

# Specialized layers using partial
StyleConv3DVel = partial(StyleBase3DVel, kernel_size=3, stride=1)
StyleSkip3DVel = partial(StyleBase3DVel, kernel_size=1, stride=1)
StyleDownSample3DVel = partial(StyleBase3DVel, kernel_size=2, stride=2)
StyleUpSample3DVel = StyleTransposeBase3DVel  # No partial needed, defaults are correct
