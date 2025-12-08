"""
Tests for nbody_emulator.py and nbody_emulator_vel.py modules.

Copyright (C) 2025 Drew Jamieson
Licensed under GNU GPL v3.0 - see LICENSE file for details.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random

from jax_nbody_emulator import (
    NBodyEmulator, 
    NBodyEmulatorVel, 
    load_default_parameters
)


class TestLoadDefaultParameters:
    """Test the parameter loading function"""
    
    def test_load_parameters_exists(self):
        """Test that default parameters can be loaded"""
        data = load_default_parameters()
        
        assert 'params' in data
        assert data['params'] is not None
    
    def test_parameters_structure(self):
        """Test that loaded parameters have correct structure"""
        data = load_default_parameters()
        params = data['params']
        
        # Should be a nested dictionary structure
        assert isinstance(params, dict)
        
        # Count total number of arrays
        leaves = jax.tree_util.tree_leaves(params)
        assert len(leaves) > 0
        
        # All leaves should be JAX arrays
        for leaf in leaves:
            assert isinstance(leaf, jnp.ndarray)
    
    def test_default_dtype_fp32(self):
        """Test that default dtype is FP32"""
        data = load_default_parameters()
        params = data['params']
        
        # Check dtypes of all parameters
        dtypes = set()
        for leaf in jax.tree_util.tree_leaves(params):
            if hasattr(leaf, 'dtype'):
                dtypes.add(leaf.dtype)
        
        # Should only have float32 (compare as numpy dtypes)
        assert dtypes == {np.dtype('float32')}

    def test_dtype_conversion_fp16(self):
        """Test conversion to FP16"""
        data = load_default_parameters(dtype=jnp.float16)
        params = data['params']
        
        # Check dtypes of all parameters
        dtypes = set()
        for leaf in jax.tree_util.tree_leaves(params):
            if hasattr(leaf, 'dtype'):
                dtypes.add(leaf.dtype)
        
        # Should only have float16
        assert dtypes == {np.dtype('float16')}

    def test_dtype_conversion_bfloat16(self):
        """Test conversion to BF16"""
        data = load_default_parameters(dtype=jnp.bfloat16)
        params = data['params']
        
        # Check dtypes of all parameters
        dtypes = set()
        for leaf in jax.tree_util.tree_leaves(params):
            if hasattr(leaf, 'dtype'):
                dtypes.add(leaf.dtype)
        
        # Should only have bfloat16
        assert dtypes == {np.dtype('bfloat16')}
 
    def test_parameters_shape_preserved(self):
        """Test that dtype conversion preserves shapes"""
        data_fp32 = load_default_parameters(dtype=jnp.float32)
        data_fp16 = load_default_parameters(dtype=jnp.float16)
        
        params_fp32 = data_fp32['params']
        params_fp16 = data_fp16['params']
        
        # Get all shapes
        shapes_fp32 = [leaf.shape for leaf in jax.tree_util.tree_leaves(params_fp32)]
        shapes_fp16 = [leaf.shape for leaf in jax.tree_util.tree_leaves(params_fp16)]
        
        # Shapes should be identical
        assert shapes_fp32 == shapes_fp16
    
    def test_parameters_have_correct_structure(self):
        """Test that loaded parameters have the structure expected by the model"""
        data = load_default_parameters(dtype=jnp.float32)
        params = data['params']
        
        # Should be a nested dict (not wrapped in another 'params' key)
        assert isinstance(params, dict)
        assert 'params' not in params
        
        # Check a sample path exists
        assert 'conv_l00' in params
        assert 'skip' in params['conv_l00']
        assert 'weight' in params['conv_l00']['skip']
    
    def test_loaded_parameters_are_valid_arrays(self):
        """Test that all loaded parameters are valid JAX arrays"""
        data = load_default_parameters(dtype=jnp.float32)
        params = data['params']
        
        # All leaves should be JAX arrays
        for leaf in jax.tree_util.tree_leaves(params):
            assert isinstance(leaf, jnp.ndarray)
            assert leaf.size > 0
            assert jnp.all(jnp.isfinite(leaf))
    
    def test_fp16_parameters_are_valid(self):
        """Test that FP16 parameters load correctly"""
        data = load_default_parameters(dtype=jnp.float16)
        params = data['params']
        
        # Check first leaf is FP16
        first_leaf = jax.tree_util.tree_leaves(params)[0]
        assert first_leaf.dtype == np.dtype('float16')
        assert jnp.all(jnp.isfinite(first_leaf))
    
    def test_parameter_count(self):
        """Test that parameter count is reasonable"""
        data = load_default_parameters()
        params = data['params']
        
        # Count total parameters
        total_params = sum(leaf.size for leaf in jax.tree_util.tree_leaves(params))
        
        # Should have exactly 3354776 parameters
        assert total_params == 3354776

    def test_complete_parameter_structure(self):
        """Test complete parameter structure"""
        data = load_default_parameters()
        
        block_keys = ['conv_c', 'conv_l00', 'conv_l01', 'conv_l1', 'conv_l2', 
                      'conv_r00', 'conv_r01', 'conv_r1', 'conv_r2', 
                      'down_l0', 'down_l1', 'down_l2', 
                      'up_r0', 'up_r1', 'up_r2']
        layer_keys = {'conv': ['conv_0', 'conv_1', 'skip'], 
                      'down': ['conv_0'], 
                      'up_r': ['conv_0']}
        layer_param_keys = ['bias', 'style_bias', 'style_weight', 'weight']
        
        # Should be a dictionary with 'params' key
        assert isinstance(data, dict)
        assert 'params' in data
        
        # Check all blocks, layers, and parameters exist
        for bk in block_keys:
            assert bk in data['params'], f"Missing block: {bk}"
            data_bk = data['params'][bk]
            
            for lk in layer_keys[bk[:4]]:
                assert lk in data_bk, f"Missing layer {lk} in block {bk}"
                data_lk = data_bk[lk]
                
                for lp_key in layer_param_keys:
                    assert lp_key in data_lk, f"Missing param {lp_key} in {bk}/{lk}"


class TestNBodyEmulator:
    """Test the main NBodyEmulator model"""
    
    def test_initialization(self):
        """Test that model initializes correctly"""
        model = NBodyEmulator()
        
        assert model.style_size == 2
        assert model.in_chan == 3
        assert model.out_chan == 3
        assert model.mid_chan == 64
    
    def test_custom_channels(self):
        """Test model with custom channel configuration"""
        model = NBodyEmulator(
            in_chan=1,
            out_chan=1,
            mid_chan=32
        )
        
        assert model.in_chan == 1
        assert model.out_chan == 1
        assert model.mid_chan == 32
    
    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shape"""
        key = random.PRNGKey(42)
        model = NBodyEmulator()
        
        # Standard input: 32^3 with 48-voxel padding
        batch_size = 1
        spatial_size = 32 + 2 * 48
        x = random.normal(key, (batch_size, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        
        params = model.init(key, x, Om, Dz)
        output = model.apply(params, x, Om, Dz)
        
        # Output is cropped by 48 voxels per side: 32 + 2 * 48 - 96 = 32
        expected_shape = (batch_size, 3, 32, 32, 32)
        assert output.shape == expected_shape
    
    def test_batch_processing(self):
        """Test that model handles batched inputs correctly"""
        key = random.PRNGKey(42)
        model = NBodyEmulator()
        
        batch_size = 2
        spatial_size = 128  # Smaller for faster test
        x = random.normal(key, (batch_size, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.2, 0.3, 0.4, 0.5])
        Dz = jnp.array([0.8, 0.9, 1.0, 1.1])
        
        params = model.init(key, x, Om, Dz)
        output = model.apply(params, x, Om, Dz)
        
        # Check batch dimension is preserved
        assert output.shape[0] == batch_size
        # Output spatial: 128 - 96 = 32
        assert output.shape == (batch_size, 3, 32, 32, 32)
    
    def test_cosmology_scaling(self):
        """Test that cosmology parameters affect output"""
        key = random.PRNGKey(42)
        model = NBodyEmulator()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        
        params = model.init(key, x, jnp.array([0.3]), jnp.array([1.0]))
        
        # Test with different cosmological parameters
        output1 = model.apply(params, x, jnp.array([0.3]), jnp.array([1.0]))
        output2 = model.apply(params, x, jnp.array([0.5]), jnp.array([1.2]))
        
        # Outputs should differ when cosmology changes
        assert not jnp.allclose(output1, output2)
    
    def test_dtype_fp16(self):
        """Test model with FP16 precision"""
        key = random.PRNGKey(42)
        model = NBodyEmulator(dtype=jnp.float16)
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size)).astype(jnp.float16)
        Om = jnp.array([0.3], dtype=jnp.float16)
        Dz = jnp.array([1.0], dtype=jnp.float16)
        
        params = model.init(key, x, Om, Dz)
        output = model.apply(params, x, Om, Dz)
        
        # Check output dtype
        assert output.dtype == jnp.float16
        assert output.shape == (1, 3, 32, 32, 32)
    
    def test_residual_connection(self):
        """Test that residual connection is working"""
        key = random.PRNGKey(42)
        model = NBodyEmulator()
        
        spatial_size = 128
        # Create input with distinct pattern
        x = jnp.ones((1, 3, spatial_size, spatial_size, spatial_size)) * 0.5
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        
        params = model.init(key, x, Om, Dz)
        output = model.apply(params, x, Om, Dz)
        
        # Output should not be constant (network is working)
        assert output.std() > 0
        # Output should not be all zeros
        assert not jnp.allclose(output, 0.0)
    
    def test_jit_compilation(self):
        """Test that model can be JIT compiled"""
        key = random.PRNGKey(42)
        model = NBodyEmulator()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        
        params = model.init(key, x, Om, Dz)
        
        # JIT compile the apply function
        jitted_apply = jax.jit(model.apply)
        
        output = jitted_apply(params, x, Om, Dz)
        
        assert output.shape == (1, 3, 32, 32, 32)


class TestNBodyEmulatorVel:
    """Test the velocity version of the emulator"""
    
    def test_initialization(self):
        """Test that velocity model initializes correctly"""
        model = NBodyEmulatorVel()
        
        assert model.style_size == 2
        assert model.in_chan == 3
        assert model.out_chan == 3
        assert model.mid_chan == 64
    
    def test_forward_pass_returns_two_outputs(self):
        """Test that velocity model returns displacement and velocity"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorVel()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([100.0])
        
        params = model.init(key, x, Om, Dz, vel_fac)
        displacement, velocity = model.apply(params, x, Om, Dz, vel_fac)
        
        # Both outputs should have same shape
        expected_shape = (1, 3, 32, 32, 32)
        assert displacement.shape == expected_shape
        assert velocity.shape == expected_shape
    
    def test_velocity_scaling(self):
        """Test that velocity scales with vel_fac parameter"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorVel()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        
        params = model.init(key, x, Om, Dz, jnp.array([100.0]))
        
        # Test with different velocity factors
        _, vel1 = model.apply(params, x, Om, Dz, jnp.array([100.0]))
        _, vel2 = model.apply(params, x, Om, Dz, jnp.array([200.0]))
        
        # Velocities should differ when vel_fac changes
        assert not jnp.allclose(vel1, vel2)
    
    def test_displacement_independent_of_vel_fac(self):
        """Test that displacement is independent of vel_fac"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorVel()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        
        params = model.init(key, x, Om, Dz, jnp.array([100.0]))
        
        # Test with different velocity factors
        dis1, _ = model.apply(params, x, Om, Dz, jnp.array([100.0]))
        dis2, _ = model.apply(params, x, Om, Dz, jnp.array([200.0]))
        
        # Displacement should be the same regardless of vel_fac
        assert jnp.allclose(dis1, dis2)
    
    def test_batch_processing_with_velocity(self):
        """Test batched processing for velocity model"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorVel()
        
        batch_size = 2
        spatial_size = 128
        x = random.normal(key, (batch_size, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3, 0.4])
        Dz = jnp.array([1.0, 1.1])
        vel_fac = jnp.array([100.0, 110.0])
        
        params = model.init(key, x, Om, Dz, vel_fac)
        displacement, velocity = model.apply(params, x, Om, Dz, vel_fac)
        
        assert displacement.shape == (batch_size, 3, 32, 32, 32)
        assert velocity.shape == (batch_size, 3, 32, 32, 32)
    
    def test_dtype_fp16_with_velocity(self):
        """Test velocity model with FP16 precision"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorVel(dtype=jnp.float16)
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size)).astype(jnp.float16)
        Om = jnp.array([0.3], dtype=jnp.float16)
        Dz = jnp.array([1.0], dtype=jnp.float16)
        vel_fac = jnp.array([100.0], dtype=jnp.float16)
        
        params = model.init(key, x, Om, Dz, vel_fac)
        displacement, velocity = model.apply(params, x, Om, Dz, vel_fac)
        
        # Check output dtypes
        assert displacement.dtype == jnp.float16
        assert velocity.dtype == jnp.float16
    
    def test_velocity_not_zero(self):
        """Test that velocity output is non-zero"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorVel()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([100.0])
        
        params = model.init(key, x, Om, Dz, vel_fac)
        _, velocity = model.apply(params, x, Om, Dz, vel_fac)
        
        # Velocity should have variation
        assert velocity.std() > 0
        # Should not be all zeros
        assert not jnp.allclose(velocity, 0.0)
    
    def test_jit_compilation_with_velocity(self):
        """Test that velocity model can be JIT compiled"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorVel()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([100.0])
        
        params = model.init(key, x, Om, Dz, vel_fac)
        
        # JIT compile the apply function
        jitted_apply = jax.jit(model.apply)
        
        displacement, velocity = jitted_apply(params, x, Om, Dz, vel_fac)
        
        assert displacement.shape == (1, 3, 32, 32, 32)
        assert velocity.shape == (1, 3, 32, 32, 32)


class TestModelComparison:
    """Compare behavior between displacement-only and velocity models"""
    
    def test_displacement_consistency(self):
        """Test that displacement output is similar between models"""
        key = random.PRNGKey(42)
        
        model_dis = NBodyEmulator()
        model_vel = NBodyEmulatorVel()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([100.0])
        
        # Initialize with same key (should have similar parameters structure)
        params_dis = model_dis.init(key, x, Om, Dz)
        params_vel = model_vel.init(key, x, Om, Dz, vel_fac)
        
        # Get outputs
        dis_output = model_dis.apply(params_dis, x, Om, Dz)
        dis_from_vel, _ = model_vel.apply(params_vel, x, Om, Dz, vel_fac)
        
        # Shapes should match
        assert dis_output.shape == dis_from_vel.shape
        
        # Note: Actual values won't match exactly because parameters are 
        # initialized differently, but both should be non-trivial
        assert dis_output.std() > 0
        assert dis_from_vel.std() > 0
    
    def test_parameter_count_similar(self):
        """Test that both models have similar parameter counts"""
        key = random.PRNGKey(42)
        
        model_dis = NBodyEmulator()
        model_vel = NBodyEmulatorVel()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        
        params_dis = model_dis.init(key, x, Om, Dz)
        params_vel = model_vel.init(key, x, Om, Dz, jnp.array([100.0]))
        
        # Count parameters
        def count_params(params):
            return sum(x.size for x in jax.tree_util.tree_leaves(params))
        
        n_params_dis = count_params(params_dis)
        n_params_vel = count_params(params_vel)
        
        # Should have exactly the same number of parameters
        assert n_params_dis == n_params_vel


class TestEdgeCases:
    """Test edge cases and numerical stability"""
    
    def test_zero_input(self):
        """Test model behavior with zero input"""
        key = random.PRNGKey(42)
        model = NBodyEmulator()
        
        spatial_size = 128
        x = jnp.zeros((1, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        
        params = model.init(key, x, Om, Dz)
        output = model.apply(params, x, Om, Dz)
        
        # Should produce output (not crash)
        assert output.shape == (1, 3, 32, 32, 32)
        # Output should not be NaN
        assert not jnp.any(jnp.isnan(output))
    
    def test_extreme_cosmology_values(self):
        """Test with extreme but valid cosmology values"""
        key = random.PRNGKey(42)
        model = NBodyEmulator()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        
        # Very low matter density
        Om_low = jnp.array([0.05])
        Dz_low = jnp.array([0.5])
        
        params = model.init(key, x, Om_low, Dz_low)
        output = model.apply(params, x, Om_low, Dz_low)
        
        # Should not produce NaN or Inf
        assert jnp.all(jnp.isfinite(output))
    
    def test_numerical_stability_fp16(self):
        """Test numerical stability with FP16"""
        key = random.PRNGKey(42)
        model = NBodyEmulator(dtype=jnp.float16)
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size)).astype(jnp.float16)
        Om = jnp.array([0.3], dtype=jnp.float16)
        Dz = jnp.array([1.0], dtype=jnp.float16)
        
        params = model.init(key, x, Om, Dz)
        output = model.apply(params, x, Om, Dz)
        
        # Should not overflow to inf or underflow to zero everywhere
        assert jnp.all(jnp.isfinite(output))
        assert output.std() > 0
