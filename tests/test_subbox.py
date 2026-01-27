"""
Tests for subbox.py module.

Subbox processing utilities for handling large volumes by splitting them into
smaller overlapping subboxes for GPU processing, then reassembling results.

Copyright (C) 2025 Drew Jamieson
Licensed under GNU GPL v3.0 - see LICENSE file for details.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random

from jax_nbody_emulator.subbox import SubboxConfig, SubboxProcessor
from jax_nbody_emulator.nbody_emulator_core import NBodyEmulatorCore
from jax_nbody_emulator.nbody_emulator_vel_core import NBodyEmulatorVelCore
from jax_nbody_emulator.style_nbody_emulator_core import StyleNBodyEmulatorCore
from jax_nbody_emulator.style_nbody_emulator_vel_core import StyleNBodyEmulatorVelCore


# =============================================================================
# SubboxConfig Tests
# =============================================================================

class TestSubboxConfig:
    """Test SubboxConfig initialization and index computation"""
    
    def test_default_initialization(self):
        """Test that SubboxConfig initializes correctly with defaults"""
        config = SubboxConfig(
            size=(256, 256, 256),
            ndiv=(2, 2, 2),
        )
        
        assert config.in_chan == 3
        assert config.size == (256, 256, 256)
        assert config.ndiv == (2, 2, 2)
        assert config.padding == ((48, 48), (48, 48), (48, 48))
        assert config.dtype == jnp.float32
        assert config.output_dtype == np.float32
        assert config.NDIM == 3
        assert config.n_subboxes == 8
        assert config.crop_size == (128, 128, 128)
    
    def test_custom_initialization(self):
        """Test SubboxConfig with custom parameters"""
        config = SubboxConfig(
            size=(512, 256, 128),
            ndiv=(4, 2, 1),
            dtype=jnp.float16,
            output_dtype=np.float16,
            in_chan=1,
            padding=((32, 32), (32, 32), (32, 32))
        )
        
        assert config.in_chan == 1
        assert config.dtype == jnp.float16
        assert config.output_dtype == np.float16
        assert config.padding == ((32, 32), (32, 32), (32, 32))
        assert config.n_subboxes == 8
        assert config.crop_size == (128, 128, 128)
    
    def test_single_subbox(self):
        """Test with single subbox (no division)"""
        config = SubboxConfig(
            size=(128, 128, 128),
            ndiv=(1, 1, 1),
        )
        
        assert config.n_subboxes == 1
        assert config.crop_size == (128, 128, 128)
    
    def test_precomputed_indices(self):
        """Test that indices are precomputed during initialization"""
        config = SubboxConfig(
            size=(256, 256, 256),
            ndiv=(2, 2, 2),
        )
        
        assert len(config.all_crop_inds) == config.n_subboxes
        assert len(config.all_add_inds) == config.n_subboxes
    
    def test_anchor_computation(self):
        """Test anchor point computation for different subbox indices"""
        config = SubboxConfig(
            size=(256, 256, 256),
            ndiv=(2, 2, 2),
        )
        
        assert config._get_anchor(0) == (0, 0, 0)
        assert config._get_anchor(1) == (0, 0, 128)
        assert config._get_anchor(7) == (128, 128, 128)
    
    def test_crop_indices_shape(self):
        """Test that crop indices have correct structure"""
        config = SubboxConfig(
            size=(256, 256, 256),
            ndiv=(2, 2, 2),
        )
        
        crop_inds = config.all_crop_inds[0]
        
        assert len(crop_inds) == 4
        assert crop_inds[0] == slice(None)
    
    def test_add_indices_no_padding(self):
        """Test that add indices have no padding"""
        config = SubboxConfig(
            size=(256, 256, 256),
            ndiv=(2, 2, 2),
        )
        
        add_inds = config.all_add_inds[0]
        
        for i in range(1, 4):
            assert add_inds[i].size == config.crop_size[i-1]
    
    def test_periodic_boundary_conditions(self):
        """Test that crop indices handle periodic boundaries"""
        config = SubboxConfig(
            size=(256, 256, 256),
            ndiv=(2, 2, 2),
        )
        
        crop_inds = config.all_crop_inds[0]
        z_inds = crop_inds[1].flatten()
        
        # Should contain wrapped indices (values >= 208)
        assert np.any(z_inds >= 208)
        # Should also contain normal indices
        assert np.any(z_inds < 128)


class TestSubboxConfigIndexConsistency:
    """Test index consistency across different configurations"""
    
    def test_crop_indices_within_bounds(self):
        """Test that crop indices are valid with periodic BC"""
        config = SubboxConfig(
            size=(256, 256, 256),
            ndiv=(2, 2, 2),
        )
        
        for idx in range(config.n_subboxes):
            crop_inds = config.all_crop_inds[idx]
            
            for dim in range(1, 4):
                inds = crop_inds[dim].flatten()
                assert np.all(inds >= 0)
                assert np.all(inds < config.size[dim-1])
    
    def test_add_indices_within_bounds(self):
        """Test that add indices are within bounds"""
        config = SubboxConfig(
            size=(256, 256, 256),
            ndiv=(2, 2, 2),
        )
        
        for idx in range(config.n_subboxes):
            add_inds = config.all_add_inds[idx]
            
            for dim in range(1, 4):
                inds = add_inds[dim].flatten()
                assert np.all(inds >= 0)
                assert np.all(inds < config.size[dim-1])
    
    def test_add_indices_contiguous(self):
        """Test that add indices form contiguous blocks"""
        config = SubboxConfig(
            size=(256, 256, 256),
            ndiv=(2, 2, 2),
        )
        
        for idx in range(config.n_subboxes):
            add_inds = config.all_add_inds[idx]
            
            for dim in range(1, 4):
                inds = add_inds[dim].flatten()
                assert np.all(np.diff(inds) == 1)
    
    def test_full_coverage(self):
        """Test that all subboxes cover the full volume exactly once"""
        config = SubboxConfig(
            size=(128, 128, 128),
            ndiv=(2, 2, 2),
        )
        
        coverage = np.zeros((128, 128, 128), dtype=int)
        
        for idx in range(config.n_subboxes):
            add_inds = config.all_add_inds[idx]
            z_inds = add_inds[1].flatten()
            y_inds = add_inds[2].flatten()
            x_inds = add_inds[3].flatten()
            
            for z in z_inds:
                for y in y_inds:
                    for x in x_inds:
                        coverage[z, y, x] += 1
        
        assert np.all(coverage == 1)


# =============================================================================
# SubboxConfig Output Dtype Tests
# =============================================================================

class TestSubboxConfigOutputDtype:
    """Test SubboxConfig output_dtype parameter"""
    
    def test_default_output_dtype_is_float32(self):
        """Test that default output_dtype is float32"""
        config = SubboxConfig(
            size=(128, 128, 128),
            ndiv=(2, 2, 2),
        )
        assert config.output_dtype == np.float32
    
    def test_explicit_output_dtype_float16(self):
        """Test setting output_dtype to float16"""
        config = SubboxConfig(
            size=(128, 128, 128),
            ndiv=(2, 2, 2),
            output_dtype=np.float16,
        )
        assert config.output_dtype == np.float16
    
    def test_output_dtype_independent_of_dtype(self):
        """Test that output_dtype and dtype are independent"""
        config = SubboxConfig(
            size=(128, 128, 128),
            ndiv=(2, 2, 2),
            dtype=jnp.float16,
            output_dtype=np.float32,
        )
        assert config.dtype == jnp.float16
        assert config.output_dtype == np.float32
        
        # And vice versa
        config2 = SubboxConfig(
            size=(128, 128, 128),
            ndiv=(2, 2, 2),
            dtype=jnp.float32,
            output_dtype=np.float16,
        )
        assert config2.dtype == jnp.float32
        assert config2.output_dtype == np.float16


# =============================================================================
# SubboxProcessor Tests - NBodyEmulatorCore (premodulate=True, compute_vel=False)
# =============================================================================

class TestSubboxProcessorNBodyEmulatorCore:
    """Test SubboxProcessor with NBodyEmulatorCore"""
    
    @pytest.fixture
    def setup_processor(self):
        """Set up model and processor for testing"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        params = model.init(key, x, Dz)
        
        config = SubboxConfig(
            size=(64, 64, 64),
            ndiv=(2, 2, 2),
        )
        
        processor = SubboxProcessor(
            model=model,
            params=params,
            config=config,
        )
        
        return processor, config
    
    def test_processor_initialization(self, setup_processor):
        """Test that processor initializes correctly"""
        processor, config = setup_processor
        
        assert processor.config == config
        assert processor.premodulate == True
        assert processor.compute_vel == False
    
    def test_process_box_output_shape(self, setup_processor):
        """Test that process_box returns correct output shape"""
        processor, config = setup_processor
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        output = processor.process_box(input_box, z=0.0, Om=0.3, show_progress=False)
        
        assert output.shape == (3, 64, 64, 64)
        assert output.dtype == np.float32
    
    def test_process_box_returns_array_not_tuple(self, setup_processor):
        """Test that process_box returns array (not tuple) when compute_vel=False"""
        processor, config = setup_processor
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        output = processor.process_box(input_box, z=0.0, Om=0.3, show_progress=False)
        
        assert isinstance(output, np.ndarray)
    
    def test_process_box_finite_output(self, setup_processor):
        """Test that process_box returns finite values"""
        processor, config = setup_processor
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        output = processor.process_box(input_box, z=0.0, Om=0.3, show_progress=False)
        
        assert np.all(np.isfinite(output))
    
    def test_process_box_deterministic(self, setup_processor):
        """Test that processing is deterministic"""
        processor, config = setup_processor
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        
        output1 = processor.process_box(input_box, z=0.0, Om=0.3, show_progress=False)
        output2 = processor.process_box(input_box, z=0.0, Om=0.3, show_progress=False)
        
        assert np.allclose(output1, output2)
    
    def test_input_preserved(self, setup_processor):
        """Test that input array is not modified"""
        processor, config = setup_processor
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        input_box_copy = input_box.copy()
        
        _ = processor.process_box(input_box, z=0.0, Om=0.3, show_progress=False)
        
        assert np.array_equal(input_box, input_box_copy)
    
    def test_different_redshift_different_output(self, setup_processor):
        """Test that different redshift produces different output"""
        processor, config = setup_processor
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        
        output1 = processor.process_box(input_box, z=0.0, Om=0.3, show_progress=False)
        output2 = processor.process_box(input_box, z=1.0, Om=0.3, show_progress=False)
        
        assert not np.allclose(output1, output2)


# =============================================================================
# SubboxProcessor Tests - NBodyEmulatorVelCore (premodulate=True, compute_vel=True)
# =============================================================================

class TestSubboxProcessorNBodyEmulatorVelCore:
    """Test SubboxProcessor with NBodyEmulatorVelCore"""
    
    @pytest.fixture
    def setup_processor_vel(self):
        """Set up velocity model and processor for testing"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([1.0])
        params = model.init(key, x, Dz, vel_fac)
        
        config = SubboxConfig(
            size=(64, 64, 64),
            ndiv=(2, 2, 2),
        )
        
        processor = SubboxProcessor(
            model=model,
            params=params,
            config=config,
        )
        
        return processor, config
    
    def test_processor_vel_initialization(self, setup_processor_vel):
        """Test that velocity processor initializes correctly"""
        processor, config = setup_processor_vel
        
        assert processor.config == config
        assert processor.premodulate == True
        assert processor.compute_vel == True
    
    def test_process_box_returns_tuple(self, setup_processor_vel):
        """Test that process_box returns tuple of displacement and velocity"""
        processor, config = setup_processor_vel
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        result = processor.process_box(input_box, z=0.0, Om=0.3, show_progress=False)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_process_box_output_shapes(self, setup_processor_vel):
        """Test that both outputs have correct shapes"""
        processor, config = setup_processor_vel
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        displacement, velocity = processor.process_box(input_box, z=0.0, Om=0.3, show_progress=False)
        
        assert displacement.shape == (3, 64, 64, 64)
        assert velocity.shape == (3, 64, 64, 64)
        assert displacement.dtype == np.float32
        assert velocity.dtype == np.float32
    
    def test_process_box_finite_outputs(self, setup_processor_vel):
        """Test that both outputs are finite"""
        processor, config = setup_processor_vel
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        displacement, velocity = processor.process_box(input_box, z=0.0, Om=0.3, show_progress=False)
        
        assert np.all(np.isfinite(displacement))
        assert np.all(np.isfinite(velocity))
    
    def test_displacement_and_velocity_different(self, setup_processor_vel):
        """Test that displacement and velocity are different"""
        processor, config = setup_processor_vel
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        displacement, velocity = processor.process_box(input_box, z=0.0, Om=0.3, show_progress=False)
        
        assert not np.allclose(displacement, velocity)


# =============================================================================
# SubboxProcessor Tests - StyleNBodyEmulatorCore (premodulate=False, compute_vel=False)
# =============================================================================

class TestSubboxProcessorStyleNBodyEmulatorCore:
    """Test SubboxProcessor with StyleNBodyEmulatorCore"""
    
    @pytest.fixture
    def setup_style_processor(self):
        """Set up style model and processor for testing"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        params = model.init(key, x, Om, Dz)
        
        config = SubboxConfig(
            size=(64, 64, 64),
            ndiv=(2, 2, 2),
        )
        
        processor = SubboxProcessor(
            model=model,
            params=params,
            config=config,
        )
        
        return processor, config
    
    def test_style_processor_initialization(self, setup_style_processor):
        """Test that style processor initializes correctly"""
        processor, config = setup_style_processor
        
        assert processor.config == config
        assert processor.premodulate == False
        assert processor.compute_vel == False
    
    def test_process_box_output_shape(self, setup_style_processor):
        """Test that process_box returns correct output shape"""
        processor, config = setup_style_processor
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        output = processor.process_box(input_box, z=0.0, Om=0.3, show_progress=False)
        
        assert output.shape == (3, 64, 64, 64)
        assert output.dtype == np.float32
    
    def test_process_box_finite_output(self, setup_style_processor):
        """Test that process_box returns finite values"""
        processor, config = setup_style_processor
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        output = processor.process_box(input_box, z=0.0, Om=0.3, show_progress=False)
        
        assert np.all(np.isfinite(output))
    
    def test_om_affects_output(self, setup_style_processor):
        """Test that Om affects output (style conditioning)"""
        processor, config = setup_style_processor
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        
        output1 = processor.process_box(input_box, z=0.0, Om=0.2, show_progress=False)
        output2 = processor.process_box(input_box, z=0.0, Om=0.4, show_progress=False)
        
        assert not np.allclose(output1, output2)


# =============================================================================
# SubboxProcessor Tests - StyleNBodyEmulatorVelCore (premodulate=False, compute_vel=True)
# =============================================================================

class TestSubboxProcessorStyleNBodyEmulatorVelCore:
    """Test SubboxProcessor with StyleNBodyEmulatorVelCore"""
    
    @pytest.fixture
    def setup_style_processor_vel(self):
        """Set up style velocity model and processor for testing"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([1.0])
        params = model.init(key, x, Om, Dz, vel_fac)
        
        config = SubboxConfig(
            size=(64, 64, 64),
            ndiv=(2, 2, 2),
        )
        
        processor = SubboxProcessor(
            model=model,
            params=params,
            config=config,
        )
        
        return processor, config
    
    def test_style_processor_vel_initialization(self, setup_style_processor_vel):
        """Test that style velocity processor initializes correctly"""
        processor, config = setup_style_processor_vel
        
        assert processor.config == config
        assert processor.premodulate == False
        assert processor.compute_vel == True
    
    def test_process_box_returns_tuple(self, setup_style_processor_vel):
        """Test that process_box returns tuple"""
        processor, config = setup_style_processor_vel
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        result = processor.process_box(input_box, z=0.0, Om=0.3, show_progress=False)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_process_box_output_shapes(self, setup_style_processor_vel):
        """Test that both outputs have correct shapes"""
        processor, config = setup_style_processor_vel
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        displacement, velocity = processor.process_box(input_box, z=0.0, Om=0.3, show_progress=False)
        
        assert displacement.shape == (3, 64, 64, 64)
        assert velocity.shape == (3, 64, 64, 64)
    
    def test_process_box_finite_outputs(self, setup_style_processor_vel):
        """Test that both outputs are finite"""
        processor, config = setup_style_processor_vel
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        displacement, velocity = processor.process_box(input_box, z=0.0, Om=0.3, show_progress=False)
        
        assert np.all(np.isfinite(displacement))
        assert np.all(np.isfinite(velocity))
    
    def test_om_affects_both_outputs(self, setup_style_processor_vel):
        """Test that Om affects both displacement and velocity"""
        processor, config = setup_style_processor_vel
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        
        disp1, vel1 = processor.process_box(input_box, z=0.0, Om=0.2, show_progress=False)
        disp2, vel2 = processor.process_box(input_box, z=0.0, Om=0.4, show_progress=False)
        
        assert not np.allclose(disp1, disp2)
        assert not np.allclose(vel1, vel2)


# =============================================================================
# Dtype Tests
# =============================================================================

class TestSubboxProcessorDtypes:
    """Test dtype handling in subbox processor"""
    
    def test_fp16_processing(self):
        """Test processing with FP16 dtype"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        params = model.init(key, x, Dz)
        
        config = SubboxConfig(
            size=(64, 64, 64),
            ndiv=(2, 2, 2),
            dtype=jnp.float16,
        )
        
        processor = SubboxProcessor(
            model=model,
            params=params,
            config=config,
        )
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        output = processor.process_box(input_box, z=0.0, Om=0.3, show_progress=False)
        
        # Output should be in output_dtype (default float32)
        assert output.dtype == np.float32
        assert np.all(np.isfinite(output))
    
    def test_fp32_processing(self):
        """Test processing with FP32 dtype (default)"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        params = model.init(key, x, Dz)
        
        config = SubboxConfig(
            size=(64, 64, 64),
            ndiv=(2, 2, 2),
            dtype=jnp.float32,
        )
        
        processor = SubboxProcessor(
            model=model,
            params=params,
            config=config,
        )
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        output = processor.process_box(input_box, z=0.0, Om=0.3, show_progress=False)
        
        assert output.dtype == np.float32
        assert np.all(np.isfinite(output))


# =============================================================================
# Output Dtype Tests
# =============================================================================

class TestSubboxProcessorOutputDtype:
    """Test output_dtype handling in subbox processor"""
    
    @pytest.fixture
    def model_and_params(self):
        """Set up model and params for reuse"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        params = model.init(key, x, Dz)
        
        return model, params
    
    @pytest.fixture
    def model_and_params_vel(self):
        """Set up velocity model and params for reuse"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorVelCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        vel_fac = jnp.array([1.0])
        params = model.init(key, x, Dz, vel_fac)
        
        return model, params
    
    def test_output_dtype_float16(self, model_and_params):
        """Test that output respects output_dtype=float16"""
        model, params = model_and_params
        
        config = SubboxConfig(
            size=(64, 64, 64),
            ndiv=(2, 2, 2),
            dtype=jnp.float32,
            output_dtype=np.float16,
        )
        
        processor = SubboxProcessor(model=model, params=params, config=config)
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        output = processor.process_box(input_box, z=0.0, Om=0.3, show_progress=False)
        
        assert output.dtype == np.float16
        assert np.all(np.isfinite(output))
    
    def test_output_dtype_float32(self, model_and_params):
        """Test that output respects output_dtype=float32"""
        model, params = model_and_params
        
        config = SubboxConfig(
            size=(64, 64, 64),
            ndiv=(2, 2, 2),
            dtype=jnp.float32,
            output_dtype=np.float32,
        )
        
        processor = SubboxProcessor(model=model, params=params, config=config)
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        output = processor.process_box(input_box, z=0.0, Om=0.3, show_progress=False)
        
        assert output.dtype == np.float32
        assert np.all(np.isfinite(output))
    
    def test_output_dtype_with_fp16_compute(self, model_and_params):
        """Test output_dtype with FP16 compute dtype"""
        model, params = model_and_params
        
        config = SubboxConfig(
            size=(64, 64, 64),
            ndiv=(2, 2, 2),
            dtype=jnp.float16,
            output_dtype=np.float16,
        )
        
        processor = SubboxProcessor(model=model, params=params, config=config)
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        output = processor.process_box(input_box, z=0.0, Om=0.3, show_progress=False)
        
        assert output.dtype == np.float16
        assert np.all(np.isfinite(output))
    
    def test_output_dtype_fp16_compute_fp32_output(self, model_and_params):
        """Test FP16 compute with FP32 output"""
        model, params = model_and_params
        
        config = SubboxConfig(
            size=(64, 64, 64),
            ndiv=(2, 2, 2),
            dtype=jnp.float16,
            output_dtype=np.float32,
        )
        
        processor = SubboxProcessor(model=model, params=params, config=config)
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        output = processor.process_box(input_box, z=0.0, Om=0.3, show_progress=False)
        
        assert output.dtype == np.float32
        assert np.all(np.isfinite(output))
    
    def test_velocity_output_dtype_float16(self, model_and_params_vel):
        """Test that both displacement and velocity respect output_dtype=float16"""
        model, params = model_and_params_vel
        
        config = SubboxConfig(
            size=(64, 64, 64),
            ndiv=(2, 2, 2),
            output_dtype=np.float16,
        )
        
        processor = SubboxProcessor(model=model, params=params, config=config)
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        displacement, velocity = processor.process_box(input_box, z=0.0, Om=0.3, show_progress=False)
        
        assert displacement.dtype == np.float16
        assert velocity.dtype == np.float16
        assert np.all(np.isfinite(displacement))
        assert np.all(np.isfinite(velocity))
    
    def test_velocity_output_dtype_float32(self, model_and_params_vel):
        """Test that both displacement and velocity respect output_dtype=float32"""
        model, params = model_and_params_vel
        
        config = SubboxConfig(
            size=(64, 64, 64),
            ndiv=(2, 2, 2),
            output_dtype=np.float32,
        )
        
        processor = SubboxProcessor(model=model, params=params, config=config)
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        displacement, velocity = processor.process_box(input_box, z=0.0, Om=0.3, show_progress=False)
        
        assert displacement.dtype == np.float32
        assert velocity.dtype == np.float32
        assert np.all(np.isfinite(displacement))
        assert np.all(np.isfinite(velocity))


# =============================================================================
# Progress Bar Tests
# =============================================================================

class TestSubboxProcessorProgressBar:
    """Test progress bar functionality"""
    
    @pytest.fixture
    def setup_processor(self):
        key = random.PRNGKey(42)
        model = NBodyEmulatorCore()
        
        x = random.normal(key, (1, 3, 128, 128, 128))
        params = model.init(key, x, jnp.array([1.0]))
        
        config = SubboxConfig(
            size=(64, 64, 64),
            ndiv=(2, 2, 2),
        )
        
        processor = SubboxProcessor(
            model=model,
            params=params,
            config=config,
        )
        
        return processor
    
    def test_progress_bar_disabled(self, setup_processor):
        """Test that processing works with progress bar disabled"""
        processor = setup_processor
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        
        output = processor.process_box(input_box, z=0.0, Om=0.3, show_progress=False)
        
        assert output.shape == (3, 64, 64, 64)
    
    def test_custom_description(self, setup_processor):
        """Test that custom description works"""
        processor = setup_processor
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        
        output = processor.process_box(
            input_box, z=0.0, Om=0.3,
            desc="Custom description",
            show_progress=False
        )
        
        assert output.shape == (3, 64, 64, 64)


# =============================================================================
# Edge Cases
# =============================================================================

class TestSubboxProcessorEdgeCases:
    """Test edge cases for SubboxProcessor"""
    
    def test_single_subbox(self):
        """Test processing with single subbox"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        params = model.init(key, x, Dz)
        
        config = SubboxConfig(
            size=(64, 64, 64),
            ndiv=(1, 1, 1),
        )
        
        processor = SubboxProcessor(
            model=model,
            params=params,
            config=config,
        )
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        output = processor.process_box(input_box, z=0.0, Om=0.3, show_progress=False)
        
        assert output.shape == (3, 64, 64, 64)
        assert np.all(np.isfinite(output))
    
    def test_asymmetric_divisions(self):
        """Test processing with asymmetric divisions"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        params = model.init(key, x, Dz)
        
        config = SubboxConfig(
            size=(128, 64, 64),
            ndiv=(2, 1, 1),
        )
        
        processor = SubboxProcessor(
            model=model,
            params=params,
            config=config,
        )
        
        input_box = np.random.randn(3, 128, 64, 64).astype(np.float32)
        output = processor.process_box(input_box, z=0.0, Om=0.3, show_progress=False)
        
        assert output.shape == (3, 128, 64, 64)
        assert np.all(np.isfinite(output))
    
    def test_zero_redshift(self):
        """Test processing at z=0"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        params = model.init(key, x, Dz)
        
        config = SubboxConfig(
            size=(64, 64, 64),
            ndiv=(2, 2, 2),
        )
        
        processor = SubboxProcessor(
            model=model,
            params=params,
            config=config,
        )
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        output = processor.process_box(input_box, z=0.0, Om=0.3, show_progress=False)
        
        assert np.all(np.isfinite(output))
    
    def test_high_redshift(self):
        """Test processing at high redshift"""
        key = random.PRNGKey(42)
        model = NBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Dz = jnp.array([1.0])
        params = model.init(key, x, Dz)
        
        config = SubboxConfig(
            size=(64, 64, 64),
            ndiv=(2, 2, 2),
        )
        
        processor = SubboxProcessor(
            model=model,
            params=params,
            config=config,
        )
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        output = processor.process_box(input_box, z=2.0, Om=0.3, show_progress=False)
        
        assert np.all(np.isfinite(output))
    
    def test_extreme_cosmology(self):
        """Test processing with extreme cosmology values"""
        key = random.PRNGKey(42)
        model = StyleNBodyEmulatorCore()
        
        spatial_size = 128
        x = random.normal(key, (1, 3, spatial_size, spatial_size, spatial_size))
        Om = jnp.array([0.3])
        Dz = jnp.array([1.0])
        params = model.init(key, x, Om, Dz)
        
        config = SubboxConfig(
            size=(64, 64, 64),
            ndiv=(2, 2, 2),
        )
        
        processor = SubboxProcessor(
            model=model,
            params=params,
            config=config,
        )
        
        input_box = np.random.randn(3, 64, 64, 64).astype(np.float32)
        
        # Test with low Om
        output_low = processor.process_box(input_box, z=0.5, Om=0.1, show_progress=False)
        assert np.all(np.isfinite(output_low))
        
        # Test with high Om
        output_high = processor.process_box(input_box, z=0.5, Om=0.5, show_progress=False)
        assert np.all(np.isfinite(output_high))
