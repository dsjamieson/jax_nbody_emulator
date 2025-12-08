"""
Tests for style_layers.py module.

Copyright (C) 2025 Drew Jamieson
Licensed under GNU GPL v3.0 - see LICENSE file for details.
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn

from jax_nbody_emulator.style_layers import (
    StyleBase3D, StyleConv3D, StyleSkip3D, StyleDownSample3D, 
    StyleUpSample3D, LeakyReLUStyled
)
from jax_nbody_emulator.style_layers_vel import (
    StyleBase3DVel, StyleConv3DVel, StyleSkip3DVel, StyleDownSample3DVel,
    StyleUpSample3DVel, LeakyReLUStyledVel
)


class TestStyleBase3D:
    """Test the base StyleBase3D class"""
    
    def test_forward_pass_shapes_batched(self):
        """Test that forward pass produces correct output shapes"""
        key = random.PRNGKey(42)
        batch_size = 2
        
        layer = StyleBase3D(in_chan=32, out_chan=64, kernel_size=3, stride=1)
        
        # Initialize parameters with batched input
        x = random.normal(key, (batch_size, 32, 8, 16, 16))
        s = random.normal(key, (batch_size, 2))
        params = layer.init(key, x, s)
        
        # Forward pass
        output = layer.apply(params, x, s)
        
        # Check output shape (VALID padding reduces spatial dims by kernel_size-1)
        expected_shape = (batch_size, 64, 6, 14, 14)
        assert output.shape == expected_shape
    
    def test_forward_pass_shapes_unbatched(self):
        """Test forward pass with unbatched input"""
        key = random.PRNGKey(42)
        
        layer = StyleBase3D(in_chan=16, out_chan=32, kernel_size=3, stride=1)
        
        # Unbatched input
        x = random.normal(key, (16, 8, 8, 8))
        s = random.normal(key, (2,))
        params = layer.init(key, x, s)
        
        output = layer.apply(params, x, s)
        
        # Output should also be unbatched
        expected_shape = (32, 6, 6, 6)
        assert output.shape == expected_shape
    
    def test_style_conditioning_effect(self):
        """Test that different style vectors produce different outputs"""
        key = random.PRNGKey(42)
        
        layer = StyleBase3D(in_chan=16, out_chan=32, kernel_size=3)
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        s1 = jnp.array([[0.3, 1.0]])
        s2 = jnp.array([[0.5, 1.5]])
        
        params = layer.init(key, x, s1)
        
        output1 = layer.apply(params, x, s1)
        output2 = layer.apply(params, x, s2)
        
        # Outputs should be different when style vectors differ
        assert not jnp.allclose(output1, output2)
    
    def test_dtype_parameter(self):
        """Test that dtype parameter works correctly"""
        key = random.PRNGKey(42)
        
        layer_fp32 = StyleBase3D(in_chan=16, out_chan=32, dtype=jnp.float32)
        layer_fp16 = StyleBase3D(in_chan=16, out_chan=32, dtype=jnp.float16)
        
        x_fp32 = random.normal(key, (1, 16, 8, 8, 8))
        s_fp32 = jnp.array([[0.3, 1.0]])
        
        params_fp32 = layer_fp32.init(key, x_fp32, s_fp32)
        params_fp16 = layer_fp16.init(key, x_fp32.astype(jnp.float16), s_fp32.astype(jnp.float16))
        
        # Check parameter dtypes
        assert params_fp32['params']['weight'].dtype == jnp.float32
        assert params_fp16['params']['weight'].dtype == jnp.float16


class TestStyleConv3D:
    """Test StyleConv3D layer"""
    
    def test_default_parameters(self):
        """Test that StyleConv3D has correct default parameters"""
        layer = StyleConv3D(in_chan=32, out_chan=64)
        
        assert layer.kernel_size == 3
        assert layer.stride == 1
    
    def test_forward_pass(self):
        """Test StyleConv3D forward pass"""
        key = random.PRNGKey(42)
        
        layer = StyleConv3D(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 8, 16, 16))
        s = jnp.array([[0.3, 1.0]])
        
        params = layer.init(key, x, s)
        output = layer.apply(params, x, s)
        
        # VALID padding: out = in - (kernel_size - 1)
        expected_shape = (1, 32, 6, 14, 14)
        assert output.shape == expected_shape


class TestStyleSkip3D:
    """Test StyleSkip3D layer (1x1x1 convolution)"""
    
    def test_default_parameters(self):
        """Test that StyleSkip3D has correct default parameters"""
        layer = StyleSkip3D(in_chan=32, out_chan=64)
        
        assert layer.kernel_size == 1
        assert layer.stride == 1
    
    def test_preserves_spatial_dimensions(self):
        """Test that StyleSkip3D preserves spatial dimensions"""
        key = random.PRNGKey(42)
        
        layer = StyleSkip3D(in_chan=32, out_chan=64)
        
        spatial_shape = (8, 16, 16)
        x = random.normal(key, (1, 32, *spatial_shape))
        s = jnp.array([[0.3, 1.0]])
        
        params = layer.init(key, x, s)
        output = layer.apply(params, x, s)
        
        # 1x1x1 kernel preserves spatial dimensions
        expected_shape = (1, 64, *spatial_shape)
        assert output.shape == expected_shape


class TestStyleDownSample3D:
    """Test StyleDownSample3D layer"""
    
    def test_default_parameters(self):
        """Test that StyleDownSample3D has correct default parameters"""
        layer = StyleDownSample3D(in_chan=32, out_chan=64)
        
        assert layer.kernel_size == 2
        assert layer.stride == 2
    
    def test_downsampling_shape(self):
        """Test that downsampling halves spatial dimensions"""
        key = random.PRNGKey(42)
        
        layer = StyleDownSample3D(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 8, 16, 16))
        s = jnp.array([[0.3, 1.0]])
        
        params = layer.init(key, x, s)
        output = layer.apply(params, x, s)
        
        # Should halve each dimension: (8,16,16) -> (4,8,8)
        expected_shape = (1, 32, 4, 8, 8)
        assert output.shape == expected_shape


class TestStyleUpSample3D:
    """Test StyleUpSample3D layer (transposed convolution)"""
    
    def test_default_parameters(self):
        """Test that StyleUpSample3D has correct default parameters"""
        layer = StyleUpSample3D(in_chan=32, out_chan=64)
        
        assert layer.kernel_size == 2
        assert layer.stride == 1
    
    def test_upsampling_shape(self):
        """Test that upsampling doubles spatial dimensions"""
        key = random.PRNGKey(42)
        
        layer = StyleUpSample3D(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 4, 8, 8))
        s = jnp.array([[0.3, 1.0]])
        
        params = layer.init(key, x, s)
        output = layer.apply(params, x, s)
        
        # Should double each dimension: (4,8,8) -> (8,16,16)
        expected_shape = (1, 32, 8, 16, 16)
        assert output.shape == expected_shape


class TestLeakyReLUStyled:
    """Test LeakyReLUStyled activation function"""
    
    def test_default_negative_slope(self):
        """Test that default negative slope is 0.01"""
        layer = LeakyReLUStyled()
        assert layer.negative_slope == 0.01
    
    def test_forward_pass(self):
        """Test forward pass"""
        key = random.PRNGKey(42)
        layer = LeakyReLUStyled()
        
        x = jnp.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])
        s = jnp.array([[0.3, 1.0]])
        
        params = layer.init(key, x, s)
        output = layer.apply(params, x, s)
        
        # Test leaky ReLU behavior
        expected = jnp.array([[-0.02, -0.01, 0.0, 1.0, 2.0]])
        assert jnp.allclose(output, expected)


class TestStyleBase3DVel:
    """Test the velocity version with tangent computation"""
    
    def test_forward_pass_with_tangent(self):
        """Test forward pass returns both output and tangent"""
        key = random.PRNGKey(42)
        
        layer = StyleBase3DVel(in_chan=16, out_chan=32, kernel_size=3)
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        s = jnp.array([[0.3, 1.0]])
        dx = random.normal(key, (1, 16, 8, 8, 8))
        
        params = layer.init(key, x, s, dx)
        
        y, dy = layer.apply(params, x, s, dx)
        
        # Check shapes
        expected_shape = (1, 32, 6, 6, 6)
        assert y.shape == expected_shape
        assert dy.shape == expected_shape
    
    def test_first_layer_no_tangent(self):
        """Test first layer behavior (dx=None)"""
        key = random.PRNGKey(42)
        
        layer = StyleBase3DVel(in_chan=16, out_chan=32, kernel_size=3)
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        s = jnp.array([[0.3, 1.0]])
        
        params = layer.init(key, x, s, None)
        
        y, dy = layer.apply(params, x, s, None)
        
        # Both should have valid shapes
        expected_shape = (1, 32, 6, 6, 6)
        assert y.shape == expected_shape
        assert dy.shape == expected_shape
    
    def test_tangent_propagation(self):
        """Test that tangents propagate correctly through layers"""
        key = random.PRNGKey(42)
        
        layer1 = StyleBase3DVel(in_chan=16, out_chan=32, kernel_size=3)
        layer2 = StyleBase3DVel(in_chan=32, out_chan=64, kernel_size=3)
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        s = jnp.array([[0.3, 1.0]])
        
        params1 = layer1.init(key, x, s, None)
        params2_key = random.fold_in(key, 1)
        
        # First layer
        y1, dy1 = layer1.apply(params1, x, s, None)
        
        # Second layer (with tangent from first)
        params2 = layer2.init(params2_key, y1, s, dy1)
        y2, dy2 = layer2.apply(params2, y1, s, dy1)
        
        # Check shapes are consistent
        assert y2.shape[1] == 64  # out_chan
        assert dy2.shape == y2.shape


class TestLeakyReLUStyledVel:
    """Test velocity version of LeakyReLU"""
    
    def test_tangent_computation(self):
        """Test that tangent is computed correctly"""
        key = random.PRNGKey(42)
        layer = LeakyReLUStyledVel(negative_slope=0.1)
        
        x = jnp.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])
        s = jnp.array([[0.3, 1.0]])
        dx = jnp.ones_like(x)
        
        params = layer.init(key, x, s, dx)
        y, dy = layer.apply(params, x, s, dx)
        
        # Check output
        expected_y = jnp.array([[-0.2, -0.1, 0.0, 1.0, 2.0]])
        assert jnp.allclose(y, expected_y)
        
        # Check tangent (slope for negative, 1.0 for positive)
        expected_dy = jnp.array([[0.1, 0.1, 0.1, 1.0, 1.0]])
        assert jnp.allclose(dy, expected_dy)


class TestParameterInitialization:
    """Test parameter initialization in style layers"""
    
    def test_weight_initialization_shapes(self):
        """Test that weights are initialized with correct shapes"""
        key = random.PRNGKey(42)
        layer = StyleConv3D(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        s = jnp.array([[0.3, 1.0]])
        
        params = layer.init(key, x, s)
        
        # Check parameter shapes
        assert params['params']['weight'].shape == (32, 16, 3, 3, 3)
        assert params['params']['bias'].shape == (32,)
        assert params['params']['style_weight'].shape == (16, 2)
        assert params['params']['style_bias'].shape == (16,)
    
    def test_style_bias_initialization(self):
        """Test that style bias is initialized to ones"""
        key = random.PRNGKey(42)
        layer = StyleConv3D(in_chan=16, out_chan=32)
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        s = jnp.array([[0.3, 1.0]])
        
        params = layer.init(key, x, s)
        style_bias = params['params']['style_bias']
        
        # Style bias should be initialized to ones
        assert jnp.allclose(style_bias, jnp.ones(16))


class TestDownUpSamplingChain:
    """Test integration between downsampling and upsampling"""
    
    def test_symmetric_down_up(self):
        """Test that down then up returns to original spatial size"""
        key = random.PRNGKey(42)
        
        down_layer = StyleDownSample3D(in_chan=16, out_chan=32)
        up_layer = StyleUpSample3D(in_chan=32, out_chan=16)
        
        spatial_shape = (8, 16, 16)
        x = random.normal(key, (1, 16, *spatial_shape))
        s = jnp.array([[0.3, 1.0]])
        
        # Downsample
        down_params = down_layer.init(key, x, s)
        down_output = down_layer.apply(down_params, x, s)
        
        # Upsample
        up_key = random.fold_in(key, 1)
        up_params = up_layer.init(up_key, down_output, s)
        up_output = up_layer.apply(up_params, down_output, s)
        
        # Should return to original spatial dimensions
        assert up_output.shape == (1, 16, *spatial_shape)
