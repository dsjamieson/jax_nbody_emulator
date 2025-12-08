"""
Tests for style_layers.py module.
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as random
from flax import linen as nn

from nbody_emulator.style_layers import (
    StyleBase3D, StyleConv3D, StyleSkip3D, StyleDownSample3D, 
    StyleUpSample3D, LeakyReLUStyled
)


class TestStyleBase3D:
    """Test the base StyleBase3D class"""
    
    def test_init_parameters(self):
        """Test that StyleBase3D initializes correctly with proper shapes"""
        spatial_in_shape = (8, 16, 16)
        layer = StyleBase3D(
            style_size=128,
            in_chan=32,
            out_chan=64,
            spatial_in_shape=spatial_in_shape,
            batch_size=2,
            kernel_size=3,
            stride=1,
            spatial_out_shape=(6, 14, 14),
            transpose=False,
            padding='VALID'
        )
        
        # Test that computed shapes are correct
        assert layer.kernel_shape == (3, 3, 3)
        assert layer.strides == (1, 1, 1)
        assert layer.conv_dims == ('NCDHW', 'OIDHW', 'NCDHW')
        assert layer.weight_shape == (64, 32, 3, 3, 3)
        assert layer.style_reshape == (2, 1, 32, 1, 1, 1)
        assert layer.fan_in_dim == (2, 3, 4, 5)
    
    def test_init_parameters_transpose(self):
        """Test StyleBase3D initialization for transpose convolution"""
        spatial_in_shape = (4, 8, 8)
        layer = StyleBase3D(
            style_size=128,
            in_chan=32,
            out_chan=16,
            spatial_in_shape=spatial_in_shape,
            batch_size=2,
            kernel_size=2,
            stride=2,
            spatial_out_shape=(8, 16, 16),
            transpose=True,
            padding='SAME'
        )
        
        # Test transpose-specific shapes
        assert layer.weight_shape == (32, 16, 2, 2, 2)
        assert layer.style_reshape == (2, 32, 1, 1, 1, 1)
        assert layer.fan_in_dim == (1, 3, 4, 5)
    
    def test_forward_pass_shapes(self):
        """Test that forward pass produces correct output shapes"""
        key = random.PRNGKey(42)
        spatial_in_shape = (8, 16, 16)
        batch_size = 2
        
        layer = StyleBase3D(
            style_size=128,
            in_chan=32,
            out_chan=64,
            spatial_in_shape=spatial_in_shape,
            batch_size=batch_size,
            kernel_size=3,
            stride=1,
            spatial_out_shape=(6, 14, 14),
            transpose=False,
            padding='VALID'
        )
        
        # Initialize parameters
        x = jnp.ones((batch_size, 32, *spatial_in_shape))
        s = jnp.ones((batch_size, 128))
        params = layer.init(key, x, s)
        
        # Forward pass
        output = layer.apply(params, x, s)
        
        # Check output shape
        expected_shape = (batch_size, 64, 6, 14, 14)
        assert output.shape == expected_shape
    
    def test_style_conditioning_effect(self):
        """Test that different style vectors produce different outputs"""
        key = random.PRNGKey(42)
        spatial_in_shape = (4, 8, 8)
        batch_size = 1
        
        layer = StyleBase3D(
            style_size=64,
            in_chan=16,
            out_chan=32,
            spatial_in_shape=spatial_in_shape,
            batch_size=batch_size,
            kernel_size=3,
            stride=1,
            spatial_out_shape=(2, 6, 6),
            transpose=False,
            padding='VALID'
        )
        
        x = jnp.ones((batch_size, 16, *spatial_in_shape))
        s1 = jnp.ones((batch_size, 64))
        s2 = jnp.ones((batch_size, 64)) * 2.0
        
        params = layer.init(key, x, s1)
        
        output1 = layer.apply(params, x, s1)
        output2 = layer.apply(params, x, s2)
        
        # Outputs should be different when style vectors differ
        assert not jnp.allclose(output1, output2)


class TestStyleConv3D:
    """Test StyleConv3D layer"""
    
    def test_default_parameters(self):
        """Test that StyleConv3D has correct default parameters"""
        layer = StyleConv3D(
            style_size=128,
            in_chan=32,
            out_chan=64,
            spatial_in_shape=(8, 16, 16)
        )
        
        assert layer.kernel_size == 3
        assert layer.stride == 1
        assert layer.transpose == False
        assert layer.padding == 'VALID'
        assert layer.batch_size == 1
    
    def test_output_shape_calculation(self):
        """Test that spatial output shape is calculated correctly"""
        spatial_in_shape = (10, 20, 20)
        layer = StyleConv3D(
            style_size=128,
            in_chan=32,
            out_chan=64,
            spatial_in_shape=spatial_in_shape
        )
        
        # For 3x3x3 kernel with stride 1 and VALID padding: out = in - 2
        expected_shape = (8, 18, 18)
        assert layer.spatial_out_shape == expected_shape
    
    def test_forward_pass(self):
        """Test StyleConv3D forward pass"""
        key = random.PRNGKey(42)
        spatial_in_shape = (8, 16, 16)
        
        layer = StyleConv3D(
            style_size=64,
            in_chan=16,
            out_chan=32,
            spatial_in_shape=spatial_in_shape
        )
        
        x = random.normal(key, (1, 16, *spatial_in_shape))
        s = random.normal(key, (1, 64))
        
        params = layer.init(key, x, s)
        output = layer.apply(params, x, s)
        
        expected_shape = (1, 32, 6, 14, 14)
        assert output.shape == expected_shape


class TestStyleSkip3D:
    """Test StyleSkip3D layer"""
    
    def test_default_parameters(self):
        """Test that StyleSkip3D has correct default parameters"""
        layer = StyleSkip3D(
            style_size=128,
            in_chan=32,
            out_chan=64,
            spatial_in_shape=(8, 16, 16)
        )
        
        assert layer.kernel_size == 1
        assert layer.stride == 1
        assert layer.transpose == False
        assert layer.padding == 'VALID'
    
    def test_preserves_spatial_dimensions(self):
        """Test that StyleSkip3D preserves spatial dimensions"""
        spatial_in_shape = (8, 16, 16)
        layer = StyleSkip3D(
            style_size=128,
            in_chan=32,
            out_chan=64,
            spatial_in_shape=spatial_in_shape
        )
        
        # 1x1x1 kernel should preserve spatial dimensions
        assert layer.spatial_out_shape == spatial_in_shape
    
    def test_forward_pass(self):
        """Test StyleSkip3D forward pass"""
        key = random.PRNGKey(42)
        spatial_in_shape = (4, 8, 8)
        
        layer = StyleSkip3D(
            style_size=64,
            in_chan=16,
            out_chan=32,
            spatial_in_shape=spatial_in_shape
        )
        
        x = random.normal(key, (1, 16, *spatial_in_shape))
        s = random.normal(key, (1, 64))
        
        params = layer.init(key, x, s)
        output = layer.apply(params, x, s)
        
        # Spatial dimensions should be preserved
        expected_shape = (1, 32, *spatial_in_shape)
        assert output.shape == expected_shape


class TestStyleDownSample3D:
    """Test StyleDownSample3D layer"""
    
    def test_default_parameters(self):
        """Test that StyleDownSample3D has correct default parameters"""
        layer = StyleDownSample3D(
            style_size=128,
            in_chan=32,
            out_chan=64,
            spatial_in_shape=(8, 16, 16)
        )
        
        assert layer.kernel_size == 2
        assert layer.stride == 2
        assert layer.transpose == False
        assert layer.padding == 'VALID'
    
    def test_downsampling_shape(self):
        """Test that downsampling halves spatial dimensions"""
        spatial_in_shape = (8, 16, 16)
        layer = StyleDownSample3D(
            style_size=128,
            in_chan=32,
            out_chan=64,
            spatial_in_shape=spatial_in_shape
        )
        
        # Should halve each dimension
        expected_shape = (4, 8, 8)
        assert layer.spatial_out_shape == expected_shape
    
    def test_forward_pass(self):
        """Test StyleDownSample3D forward pass"""
        key = random.PRNGKey(42)
        spatial_in_shape = (8, 16, 16)
        
        layer = StyleDownSample3D(
            style_size=64,
            in_chan=16,
            out_chan=32,
            spatial_in_shape=spatial_in_shape
        )
        
        x = random.normal(key, (1, 16, *spatial_in_shape))
        s = random.normal(key, (1, 64))
        
        params = layer.init(key, x, s)
        output = layer.apply(params, x, s)
        
        expected_shape = (1, 32, 4, 8, 8)
        assert output.shape == expected_shape


class TestStyleUpSample3D:
    """Test StyleUpSample3D layer"""
    
    def test_default_parameters(self):
        """Test that StyleUpSample3D has correct default parameters"""
        layer = StyleUpSample3D(
            style_size=128,
            in_chan=32,
            out_chan=64,
            spatial_in_shape=(4, 8, 8)
        )
        
        assert layer.kernel_size == 2
        assert layer.stride == 2
        assert layer.transpose == True
        assert layer.padding == 'SAME'
    
    def test_upsampling_shape(self):
        """Test that upsampling doubles spatial dimensions"""
        spatial_in_shape = (4, 8, 8)
        layer = StyleUpSample3D(
            style_size=128,
            in_chan=32,
            out_chan=64,
            spatial_in_shape=spatial_in_shape
        )
        
        # Should double each dimension
        expected_shape = (8, 16, 16)
        assert layer.spatial_out_shape == expected_shape
    
    def test_forward_pass(self):
        """Test StyleUpSample3D forward pass"""
        key = random.PRNGKey(42)
        spatial_in_shape = (4, 8, 8)
        
        layer = StyleUpSample3D(
            style_size=64,
            in_chan=16,
            out_chan=32,
            spatial_in_shape=spatial_in_shape
        )
        
        x = random.normal(key, (1, 16, *spatial_in_shape))
        s = random.normal(key, (1, 64))
        
        params = layer.init(key, x, s)
        output = layer.apply(params, x, s)
        
        expected_shape = (1, 32, 8, 16, 16)
        assert output.shape == expected_shape


class TestLeakyReLUStyled:
    """Test LeakyReLUStyled activation function"""
    
    def test_default_negative_slope(self):
        """Test that default negative slope is 0.01"""
        layer = LeakyReLUStyled()
        assert layer.negative_slope == 0.01
    
    def test_custom_negative_slope(self):
        """Test setting custom negative slope"""
        layer = LeakyReLUStyled(negative_slope=0.1)
        assert layer.negative_slope == 0.1
    
    def test_forward_pass_no_style(self):
        """Test forward pass without style vector"""
        key = random.PRNGKey(42)
        layer = LeakyReLUStyled()
        
        x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        params = layer.init(key, x)
        output = layer.apply(params, x)
        
        # Test leaky ReLU behavior
        expected = jnp.array([-0.02, -0.01, 0.0, 1.0, 2.0])
        assert jnp.allclose(output, expected)
    
    def test_forward_pass_with_style(self):
        """Test forward pass with style vector (should be ignored)"""
        key = random.PRNGKey(42)
        layer = LeakyReLUStyled()
        
        x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        s = jnp.array([1.0, 2.0, 3.0])  # Should be ignored
        
        params = layer.init(key, x, s)
        output = layer.apply(params, x, s)
        
        # Should behave identically to no style case
        expected = jnp.array([-0.02, -0.01, 0.0, 1.0, 2.0])
        assert jnp.allclose(output, expected)
    
    def test_multidimensional_input(self):
        """Test LeakyReLUStyled with multidimensional input"""
        key = random.PRNGKey(42)
        layer = LeakyReLUStyled(negative_slope=0.1)
        
        x = jnp.array([[[[-1.0, 2.0], [0.0, -3.0]]]])
        params = layer.init(key, x)
        output = layer.apply(params, x)
        
        expected = jnp.array([[[[-0.1, 2.0], [0.0, -0.3]]]])
        assert jnp.allclose(output, expected)


class TestStyleLayersIntegration:
    """Test integration between different style layers"""
    
    def test_downsampling_upsampling_chain(self):
        """Test that downsampling followed by upsampling works correctly"""
        key = random.PRNGKey(42)
        spatial_in_shape = (8, 16, 16)
        
        # Create downsampling layer
        down_layer = StyleDownSample3D(
            style_size=64,
            in_chan=16,
            out_chan=32,
            spatial_in_shape=spatial_in_shape
        )
        
        # Create upsampling layer with matching dimensions
        up_layer = StyleUpSample3D(
            style_size=64,
            in_chan=32,
            out_chan=16,
            spatial_in_shape=down_layer.spatial_out_shape
        )
        
        x = random.normal(key, (1, 16, *spatial_in_shape))
        s = random.normal(key, (1, 64))
        
        # Initialize both layers
        down_params = down_layer.init(key, x, s)
        down_output = down_layer.apply(down_params, x, s)
        
        up_params = up_layer.init(key, down_output, s)
        up_output = up_layer.apply(up_params, down_output, s)
        
        # Final output should have same spatial dimensions as input
        assert up_output.shape == (1, 16, *spatial_in_shape)
    
    def test_style_consistency_across_layers(self):
        """Test that the same style vector affects different layers consistently"""
        key = random.PRNGKey(42)
        spatial_in_shape = (4, 8, 8)
        
        conv_layer = StyleConv3D(
            style_size=64,
            in_chan=16,
            out_chan=32,
            spatial_in_shape=spatial_in_shape
        )
        
        skip_layer = StyleSkip3D(
            style_size=64,
            in_chan=16,
            out_chan=32,
            spatial_in_shape=spatial_in_shape
        )
        
        x = random.normal(key, (1, 16, *spatial_in_shape))
        s = random.normal(key, (1, 64))
        
        # Test that layers can be initialized with same style vector
        conv_params = conv_layer.init(key, x, s)
        skip_params = skip_layer.init(key, x, s)
        
        conv_output = conv_layer.apply(conv_params, x, s)
        skip_output = skip_layer.apply(skip_params, x, s)
        
        # Outputs should have expected shapes
        assert conv_output.shape == (1, 32, 2, 6, 6)
        assert skip_output.shape == (1, 32, 4, 8, 8)
    
    def test_activation_layer_compatibility(self):
        """Test that activation layer works with style layer outputs"""
        key = random.PRNGKey(42)
        spatial_in_shape = (4, 8, 8)
        
        conv_layer = StyleConv3D(
            style_size=64,
            in_chan=16,
            out_chan=32,
            spatial_in_shape=spatial_in_shape
        )
        
        activation = LeakyReLUStyled(negative_slope=0.1)
        
        x = random.normal(key, (1, 16, *spatial_in_shape))
        s = random.normal(key, (1, 64))
        
        # Forward pass through conv layer
        conv_params = conv_layer.init(key, x, s)
        conv_output = conv_layer.apply(conv_params, x, s)
        
        # Forward pass through activation
        act_params = activation.init(key, conv_output, s)
        final_output = activation.apply(act_params, conv_output, s)
        
        # Shape should be preserved
        assert final_output.shape == conv_output.shape
        
        # Activation should have been applied
        assert not jnp.allclose(final_output, conv_output)


class TestParameterInitialization:
    """Test parameter initialization in style layers"""
    
    def test_weight_initialization_shapes(self):
        """Test that weights are initialized with correct shapes"""
        key = random.PRNGKey(42)
        layer = StyleConv3D(
            style_size=64,
            in_chan=16,
            out_chan=32,
            spatial_in_shape=(4, 8, 8)
        )
        
        x = jnp.ones((1, 16, 4, 8, 8))
        s = jnp.ones((1, 64))
        
        params = layer.init(key, x, s)
        
        # Check parameter shapes
        assert params['params']['weight'].shape == (32, 16, 3, 3, 3)
        assert params['params']['bias'].shape == (32,)
        assert params['params']['style_weight'].shape == (16, 64)
        assert params['params']['style_bias'].shape == (16,)
    
    def test_bias_initialization_bounds(self):
        """Test that bias is initialized within expected bounds"""
        key = random.PRNGKey(42)
        layer = StyleConv3D(
            style_size=64,
            in_chan=16,
            out_chan=32,
            spatial_in_shape=(4, 8, 8)
        )
        
        x = jnp.ones((1, 16, 4, 8, 8))
        s = jnp.ones((1, 64))
        
        params = layer.init(key, x, s)
        bias = params['params']['bias']
        
        # Bias should be initialized with uniform distribution
        # For fan_in = 16 * 3^3 = 432, bound = 1/sqrt(432) ≈ 0.048
        expected_bound = 1.0 / jnp.sqrt(16 * 27)
        assert jnp.all(jnp.abs(bias) <= expected_bound + 1e-6)
    
    def test_style_bias_initialization(self):
        """Test that style bias is initialized to ones"""
        key = random.PRNGKey(42)
        layer = StyleConv3D(
            style_size=64,
            in_chan=16,
            out_chan=32,
            spatial_in_shape=(4, 8, 8)
        )
        
        x = jnp.ones((1, 16, 4, 8, 8))
        s = jnp.ones((1, 64))
        
        params = layer.init(key, x, s)
        style_bias = params['params']['style_bias']
        
        # Style bias should be initialized to ones
        assert jnp.allclose(style_bias, jnp.ones(16))
