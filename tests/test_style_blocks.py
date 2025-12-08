"""
Tests for style_blocks.py module.

Copyright (C) 2025 Drew Jamieson
Licensed under GNU GPL v3.0 - see LICENSE file for details.
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as random

from jax_nbody_emulator.style_blocks import (
    StyleResampleBlock3D, StyleResNetBlock3D
)
from jax_nbody_emulator.style_blocks_vel import (
    StyleResampleBlock3DVel, StyleResNetBlock3DVel
)


class TestStyleResampleBlock3D:
    """Test StyleResampleBlock3D"""
    
    def test_upsampling_block(self):
        """Test block with upsampling layers"""
        key = random.PRNGKey(42)
        
        # Block: Upsample -> Activation -> Upsample
        block = StyleResampleBlock3D(
            seq='UAU',
            style_size=2,
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 4, 8, 8))
        s = jnp.array([[0.3, 1.0]])
        
        params = block.init(key, x, s)
        output = block.apply(params, x, s)
        
        # Two upsamplings: (4,8,8) -> (8,16,16) -> (16,32,32)
        expected_shape = (1, 32, 16, 32, 32)
        assert output.shape == expected_shape
    
    def test_downsampling_block(self):
        """Test block with downsampling layers"""
        key = random.PRNGKey(42)
        
        # Block: Downsample -> Activation
        block = StyleResampleBlock3D(
            seq='DA',
            style_size=2,
            in_chan=32,
            out_chan=64
        )
        
        x = random.normal(key, (1, 32, 8, 16, 16))
        s = jnp.array([[0.3, 1.0]])
        
        params = block.init(key, x, s)
        output = block.apply(params, x, s)
        
        # One downsampling: (8,16,16) -> (4,8,8)
        expected_shape = (1, 64, 4, 8, 8)
        assert output.shape == expected_shape
    
    def test_mixed_resampling(self):
        """Test block with both up and down sampling"""
        key = random.PRNGKey(42)
        
        # Block: Down -> Act -> Up -> Act
        block = StyleResampleBlock3D(
            seq='DAUA',
            style_size=2,
            in_chan=16,
            out_chan=16
        )
        
        spatial_shape = (8, 16, 16)
        x = random.normal(key, (1, 16, *spatial_shape))
        s = jnp.array([[0.3, 1.0]])
        
        params = block.init(key, x, s)
        output = block.apply(params, x, s)
        
        # Down then up: (8,16,16) -> (4,8,8) -> (8,16,16)
        expected_shape = (1, 16, *spatial_shape)
        assert output.shape == expected_shape
    
    def test_invalid_layer_type(self):
        """Test that invalid layer type raises error during initialization"""
        key = random.PRNGKey(42)
        
        block = StyleResampleBlock3D(
            seq='UXA',  # X is invalid
            style_size=2,
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 4, 8, 8))
        s = jnp.array([[0.3, 1.0]])
        
        with pytest.raises(ValueError, match='Layer type "X" not supported'):
            params = block.init(key, x, s)

    def test_dtype_parameter(self):
        """Test that dtype parameter is respected"""
        key = random.PRNGKey(42)
        
        block_fp16 = StyleResampleBlock3D(
            seq='UA',
            style_size=2,
            in_chan=16,
            out_chan=32,
            dtype=jnp.float16
        )
        
        x = random.normal(key, (1, 16, 4, 8, 8)).astype(jnp.float16)
        s = jnp.array([[0.3, 1.0]], dtype=jnp.float16)
        
        params = block_fp16.init(key, x, s)
        
        # Check that parameters are in correct dtype
        first_param = jax.tree_util.tree_leaves(params)[0]
        assert first_param.dtype == jnp.float16


class TestStyleResNetBlock3D:
    """Test StyleResNetBlock3D"""
    
    def test_basic_resnet_block(self):
        """Test basic ResNet block: Conv -> Act -> Conv"""
        key = random.PRNGKey(42)
        
        block = StyleResNetBlock3D(
            seq='CAC',
            style_size=2,
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        s = jnp.array([[0.3, 1.0]])
        
        params = block.init(key, x, s)
        output = block.apply(params, x, s)
        
        # Two 3x3x3 convs crop 2 voxels per side: (8,8,8) -> (4,4,4)
        expected_shape = (1, 32, 4, 4, 4)
        assert output.shape == expected_shape
    
    def test_resnet_with_final_activation(self):
        """Test ResNet block with final activation after residual"""
        key = random.PRNGKey(42)
        
        block = StyleResNetBlock3D(
            seq='CACA',  # Final A applies after residual addition
            style_size=2,
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        s = jnp.array([[0.3, 1.0]])
        
        params = block.init(key, x, s)
        output = block.apply(params, x, s)
        
        expected_shape = (1, 32, 4, 4, 4)
        assert output.shape == expected_shape
    
    def test_single_conv_resnet(self):
        """Test ResNet block with single convolution"""
        key = random.PRNGKey(42)
        
        block = StyleResNetBlock3D(
            seq='CA',
            style_size=2,
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        s = jnp.array([[0.3, 1.0]])
        
        params = block.init(key, x, s)
        output = block.apply(params, x, s)
        
        # One conv crops 1 voxel per side: (8,8,8) -> (6,6,6)
        expected_shape = (1, 32, 6, 6, 6)
        assert output.shape == expected_shape
    
    def test_skip_connection_cropping(self):
        """Test that skip connection is cropped correctly"""
        key = random.PRNGKey(42)
        
        # Three convolutions should crop 3 voxels per side from skip
        block = StyleResNetBlock3D(
            seq='CACAC',
            style_size=2,
            in_chan=16,
            out_chan=16  # Same channels for easier testing
        )
        
        x = random.normal(key, (1, 16, 10, 10, 10))
        s = jnp.array([[0.3, 1.0]])
        
        params = block.init(key, x, s)
        output = block.apply(params, x, s)
        
        # Three convs: (10,10,10) -> (4,4,4)
        # Skip is cropped by 3 per side: (10,10,10) -> (4,4,4)
        expected_shape = (1, 16, 4, 4, 4)
        assert output.shape == expected_shape
    
    def test_residual_identity_channels(self):
        """Test that residual connection adds both paths"""
        key = random.PRNGKey(42)
        
        block = StyleResNetBlock3D(
            seq='CA',
            style_size=2,
            in_chan=16,
            out_chan=16
        )
        
        # Use non-zero input to test residual addition
        x = random.normal(key, (1, 16, 8, 8, 8))
        s = jnp.array([[0.3, 1.0]])
        
        params = block.init(key, x, s)
        output = block.apply(params, x, s)
        
        # Output should be non-zero and have correct shape
        assert output.shape == (1, 16, 6, 6, 6)
        assert not jnp.allclose(output, 0.0)
        
        # Test that changing input changes output (residual is working)
        x2 = random.normal(random.fold_in(key, 1), (1, 16, 8, 8, 8))
        output2 = block.apply(params, x2, s)
        
        # Different inputs should give different outputs
        assert not jnp.allclose(output, output2)    

    def test_invalid_layer_type(self):
        """Test that invalid layer type raises error during initialization"""
        key = random.PRNGKey(42)
        
        block = StyleResNetBlock3D(
            seq='CXA',  # X is invalid
            style_size=2,
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        s = jnp.array([[0.3, 1.0]])
        
        # Error is raised during init
        with pytest.raises(ValueError, match='Layer type "X" not supported'):
            params = block.init(key, x, s)

class TestStyleResampleBlock3DVel:
    """Test velocity version of resample block"""
    
    def test_tangent_propagation(self):
        """Test that tangents propagate through resampling"""
        key = random.PRNGKey(42)
        
        block = StyleResampleBlock3DVel(
            seq='UA',
            style_size=2,
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 4, 8, 8))
        s = jnp.array([[0.3, 1.0]])
        dx = random.normal(key, (1, 16, 4, 8, 8))
        
        params = block.init(key, x, s, dx)
        y, dy = block.apply(params, x, s, dx)
        
        # Check both outputs have correct shape
        expected_shape = (1, 32, 8, 16, 16)
        assert y.shape == expected_shape
        assert dy.shape == expected_shape
    
    def test_first_layer_no_tangent(self):
        """Test first layer behavior (dx=None)"""
        key = random.PRNGKey(42)
        
        block = StyleResampleBlock3DVel(
            seq='DA',
            style_size=2,
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 8, 16, 16))
        s = jnp.array([[0.3, 1.0]])
        
        params = block.init(key, x, s, None)
        y, dy = block.apply(params, x, s, None)
        
        expected_shape = (1, 32, 4, 8, 8)
        assert y.shape == expected_shape
        assert dy.shape == expected_shape


class TestStyleResNetBlock3DVel:
    """Test velocity version of ResNet block"""
    
    def test_tangent_through_residual(self):
        """Test that tangents propagate through residual connection"""
        key = random.PRNGKey(42)
        
        block = StyleResNetBlock3DVel(
            seq='CAC',
            style_size=2,
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        s = jnp.array([[0.3, 1.0]])
        dx = random.normal(key, (1, 16, 8, 8, 8))
        
        params = block.init(key, x, s, dx)
        y, dy = block.apply(params, x, s, dx)
        
        expected_shape = (1, 32, 4, 4, 4)
        assert y.shape == expected_shape
        assert dy.shape == expected_shape
    
    def test_tangent_addition_in_residual(self):
        """Test that tangents are added in residual connection"""
        key = random.PRNGKey(42)
        
        block = StyleResNetBlock3DVel(
            seq='CA',
            style_size=2,
            in_chan=16,
            out_chan=16
        )
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        s = jnp.array([[0.3, 1.0]])
        dx = jnp.ones((1, 16, 8, 8, 8))
        
        params = block.init(key, x, s, dx)
        y, dy = block.apply(params, x, s, dx)
        
        # Tangent should not be all zeros or all ones (sum of two contributions)
        assert not jnp.allclose(dy, 0.0)
        assert not jnp.allclose(dy, 1.0)
    
    def test_final_activation_with_tangent(self):
        """Test final activation is applied to tangents correctly"""
        key = random.PRNGKey(42)
        
        block = StyleResNetBlock3DVel(
            seq='CACA',  # Final A after residual
            style_size=2,
            in_chan=16,
            out_chan=32
        )
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        s = jnp.array([[0.3, 1.0]])
        dx = random.normal(key, (1, 16, 8, 8, 8))
        
        params = block.init(key, x, s, dx)
        y, dy = block.apply(params, x, s, dx)
        
        expected_shape = (1, 32, 4, 4, 4)
        assert y.shape == expected_shape
        assert dy.shape == expected_shape


class TestBlockIntegration:
    """Test integration between different blocks"""
    
    def test_encoder_decoder_pipeline(self):
        """Test typical encoder-decoder architecture"""
        key = random.PRNGKey(42)
        
        # Encoder: ResNet -> Downsample
        encoder = StyleResNetBlock3D(
            seq='CAC',
            style_size=2,
            in_chan=16,
            out_chan=32
        )
        
        downsample = StyleResampleBlock3D(
            seq='DA',
            style_size=2,
            in_chan=32,
            out_chan=64
        )
        
        # Decoder: Upsample -> ResNet
        upsample = StyleResampleBlock3D(
            seq='UA',
            style_size=2,
            in_chan=64,
            out_chan=32
        )
        
        decoder = StyleResNetBlock3D(
            seq='CAC',
            style_size=2,
            in_chan=32,
            out_chan=16
        )
        
        x = random.normal(key, (1, 16, 16, 16, 16))
        s = jnp.array([[0.3, 1.0]])
        
        # Encode
        enc_params = encoder.init(key, x, s)
        x1 = encoder.apply(enc_params, x, s)  # (16,16,16) -> (12,12,12)
        
        down_params = downsample.init(random.fold_in(key, 1), x1, s)
        x2 = downsample.apply(down_params, x1, s)  # (12,12,12) -> (6,6,6)
        
        # Decode
        up_params = upsample.init(random.fold_in(key, 2), x2, s)
        x3 = upsample.apply(up_params, x2, s)  # (6,6,6) -> (12,12,12)
        
        dec_params = decoder.init(random.fold_in(key, 3), x3, s)
        x4 = decoder.apply(dec_params, x3, s)  # (12,12,12) -> (8,8,8)
        
        assert x4.shape == (1, 16, 8, 8, 8)
    
    def test_velocity_pipeline(self):
        """Test velocity blocks in pipeline"""
        key = random.PRNGKey(42)
        
        block1 = StyleResNetBlock3DVel(
            seq='CA',
            style_size=2,
            in_chan=16,
            out_chan=32
        )
        
        block2 = StyleResampleBlock3DVel(
            seq='DA',
            style_size=2,
            in_chan=32,
            out_chan=64
        )
        
        x = random.normal(key, (1, 16, 8, 8, 8))
        s = jnp.array([[0.3, 1.0]])
        
        # First block (no initial tangent)
        params1 = block1.init(key, x, s, None)
        y1, dy1 = block1.apply(params1, x, s, None)
        
        # Second block (with tangent from first)
        params2 = block2.init(random.fold_in(key, 1), y1, s, dy1)
        y2, dy2 = block2.apply(params2, y1, s, dy1)
        
        assert y2.shape == (1, 64, 3, 3, 3)
        assert dy2.shape == (1, 64, 3, 3, 3)
