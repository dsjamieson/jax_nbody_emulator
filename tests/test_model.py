import pytest
import jax
import jax.numpy as jnp
from nbody_emulator.networks import NBodyEmulator

@pytest.fixture
def model():
    return NBodyEmulator(
        style_size=2,
        in_chan=1,
        out_chan=1,
        spatial_in_shape=(1, 1, 32, 32, 32)  # Small for testing
    )

def test_model_initialization(model):
    """Test that model can be initialized"""
    key = jax.random.PRNGKey(0)
    x = jnp.ones((1, 1, 32, 32, 32))
    omega_m = jnp.array([0.3])
    growth_factor = jnp.array([1.0])
    
    params = model.init(key, x, omega_m, growth_factor)
    assert params is not None

def test_forward_pass(model):
    """Test forward pass produces correct output shape"""
    key = jax.random.PRNGKey(0)
    x = jnp.ones((1, 1, 32, 32, 32))
    omega_m = jnp.array([0.3])
    growth_factor = jnp.array([1.0])
    
    params = model.init(key, x, omega_m, growth_factor)
    output = model.apply(params, x, omega_m, growth_factor)
    
    assert output.shape == x.shape

def test_different_batch_sizes(model):
    """Test model works with different batch sizes"""
    key = jax.random.PRNGKey(0)
    
    for batch_size in [1, 2, 4]:
        x = jnp.ones((batch_size, 1, 32, 32, 32))
        omega_m = jnp.array([0.3] * batch_size)
        growth_factor = jnp.array([1.0] * batch_size)
        
        params = model.init(key, x, omega_m, growth_factor)
        output = model.apply(params, x, omega_m, growth_factor)
        assert output.shape == x.shape

def test_parameter_count(model):
    """Test that model has reasonable number of parameters"""
    key = jax.random.PRNGKey(0)
    x = jnp.ones((1, 1, 32, 32, 32))
    omega_m = jnp.array([0.3])
    growth_factor = jnp.array([1.0])
    
    params = model.init(key, x, omega_m, growth_factor)
    param_count = sum(p.size for p in jax.tree_leaves(params))
    assert param_count > 0
    assert param_count < 1e8  # Reasonable upper bound
