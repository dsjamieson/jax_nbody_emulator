"""
Tests for cosmology.py module.
"""
import pytest
import jax
import jax.numpy as jnp
from nbody_emulator.cosmology import (
    D, H, Omega_m_of_z, f, vel_norm, acc_norm, growth_acc, 
    dlogH_dloga, _growth_2f1, _log_a, _log_D, _log_H
)

class TestBasicCosmology:
    """Test basic cosmological functions"""
    
    def test_growth_factor_at_z_zero(self):
        """Growth factor should be 1 at z=0"""
        assert jnp.isclose(D(0.0, 0.3), 1.0, rtol=1e-6)
        assert jnp.isclose(D(0.0, 0.1), 1.0, rtol=1e-6)
        assert jnp.isclose(D(0.0, 0.5), 1.0, rtol=1e-6)
    
    def test_growth_factor_decreases_with_redshift(self):
        """Growth factor should decrease with increasing redshift"""
        omega_m = 0.3
        z_values = jnp.array([0.0, 0.5, 1.0, 2.0, 3.0])
        D_values = D(z_values, omega_m)
        
        # Should be monotonically decreasing
        assert jnp.all(jnp.diff(D_values) < 0)
    
    def test_hubble_parameter_at_z_zero(self):
        """Hubble parameter should equal 100 at z=0 (in units of h km/s/Mpc)"""
        assert jnp.isclose(H(0.0, 0.3), 100.0, rtol=1e-6)
        assert jnp.isclose(H(0.0, 0.1), 100.0, rtol=1e-6)
    
    def test_hubble_parameter_increases_with_redshift(self):
        """Hubble parameter should increase with redshift"""
        omega_m = 0.3
        z_values = jnp.array([0.0, 0.5, 1.0, 2.0, 3.0])
        H_values = H(z_values, omega_m)
        
        # Should be monotonically increasing
        assert jnp.all(jnp.diff(H_values) > 0)
    
    def test_growth_rate_positive(self):
        """Growth rate f should be positive"""
        omega_m = 0.3
        z_values = jnp.array([0.0, 0.5, 1.0, 2.0])
        f_values = f(z_values, omega_m)
        
        assert jnp.all(f_values > 0)
        assert jnp.all(f_values < 2.0)  # Reasonable upper bound

class TestCosmologyArrays:
    """Test that functions work with arrays"""
    
    def test_vectorized_operations(self):
        """Test that functions work with array inputs"""
        z_array = jnp.array([0.0, 0.5, 1.0, 2.0])
        omega_m_array = jnp.array([0.2, 0.3, 0.4, 0.5])
        
        # Test with array redshifts, scalar omega_m
        D_z = D(z_array, 0.3)
        assert D_z.shape == z_array.shape
        
        # Test with scalar redshift, array omega_m
        D_om = D(0.5, omega_m_array)
        assert D_om.shape == omega_m_array.shape
        
        # Test with both arrays (should broadcast)
        D_both = D(z_array, omega_m_array)
        assert D_both.shape == z_array.shape

class TestCosmologyDerivatives:
    """Test derivative functions"""
    
    def test_growth_rate_finite_difference(self):
        """Test growth rate against finite difference"""
        z = 1.0
        omega_m = 0.3
        dz = 1e-2
        
        # Finite difference approximation
        dlogD_dz_fd = (jnp.log(D(z + dz, omega_m)) - jnp.log(D(z - dz, omega_m))) / (2 * dz)
        f_fd = -dlogD_dz_fd * (1 + z)
        
        # Automatic differentiation result
        f_ad = f(z, omega_m)
        
        assert jnp.isclose(f_ad, f_fd, rtol=1e-4)
    
    def test_hubble_derivative(self):
        """Test Hubble derivative function"""
        z = 1.0
        omega_m = 0.3
        
        # Should be finite
        dlogH_dloga_val = dlogH_dloga(z, omega_m)
        assert jnp.isfinite(dlogH_dloga_val)
        
        # Should be negative (H decreases as a increases)
        assert dlogH_dloga_val < 0
    
    def test_growth_acceleration_finite(self):
        """Test growth acceleration is finite"""
        z_values = jnp.array([0.0, 0.5, 1.0, 2.0])
        omega_m = 0.3
        
        acc_values = growth_acc(z_values, omega_m)
        assert jnp.all(jnp.isfinite(acc_values))

class TestCosmologyPhysics:
    """Test physical consistency"""
    
    def test_einstein_de_sitter_limit(self):
        """Test Einstein-de Sitter limit (Om=1)"""
        omega_m = 0.99999  # Close to 1
        z = 1.0
        
        # In EdS: D(z) = 1/(1+z), f = 1
        D_eds = D(z, omega_m)
        f_eds = f(z, omega_m)
        
        expected_D = 1.0 / (1 + z)
        expected_f = 1.0
        
        assert jnp.isclose(D_eds, expected_D, rtol=1e-3)
        assert jnp.isclose(f_eds, expected_f, rtol=1e-2)
    
    def test_velocity_and_acceleration_units(self):
        """Test velocity and acceleration normalization factors"""
        z = 1.0
        omega_m = 0.3
        
        vel = vel_norm(z, omega_m)
        acc = acc_norm(z, omega_m)
        
        # Should be positive and finite
        assert vel > 0 and jnp.isfinite(vel)
        assert jnp.isfinite(acc)  # Can be positive or negative
        
        # Velocity should have units of km/s (order of magnitude check)
        assert 10 < vel < 1000  # Reasonable range
    
    def test_omega_m_dependence(self):
        """Test dependence on matter density parameter"""
        z = 1.0
        omega_m_values = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
        
        D_values = D(z, omega_m_values)
        f_values = f(z, omega_m_values)
        
        # Higher omega_m should give lower growth at fixed z if D(z=0)=1
        assert jnp.all(jnp.diff(D_values) < 0)
        # Growth rate should also increase with omega_m
        assert jnp.all(jnp.diff(f_values) > 0)

class TestCosmologyEdgeCases:
    """Test edge cases and numerical stability"""
    
    def test_high_redshift_behavior(self):
        """Test behavior at high redshift"""
        omega_m = 0.3
        z_high = jnp.array([5.0, 10.0, 20.0])
        
        D_high = D(z_high, omega_m)
        f_high = f(z_high, omega_m)
        
        # Should be finite
        assert jnp.all(jnp.isfinite(D_high))
        assert jnp.all(jnp.isfinite(f_high))
        
        # Should be small at high z
        assert jnp.all(D_high < 0.25)
        
        # f should approach omega_m^0.6 at high z
        expected_f = Omega_m_of_z(z_high, omega_m) ** 0.55
        assert jnp.all(jnp.isclose(f_high, expected_f, rtol=1.e-4))
    
    def test_invalid_inputs_produce_nan(self):
        """Test that invalid inputs produce NaN (document this behavior)"""
        # Negative redshift
        assert jnp.isnan(D(-1.0, 0.3)) or not jnp.isfinite(D(-1.0, 0.3))
        
        # Negative omega_m
        assert jnp.isnan(D(0.5, -0.1)) or not jnp.isfinite(D(0.5, -0.1))
        
    def test_very_small_omega_m(self):
        """Test behavior with very small omega_m"""
        omega_m = 1e-6
        z = 1.0
        
        # Should still be finite
        D_small = D(z, omega_m)
        f_small = f(z, omega_m)
        
        assert jnp.isfinite(D_small)
        assert jnp.isfinite(f_small)

class TestJAXCompatibility:
    """Test JAX-specific functionality"""
    
    def test_jit_compilation(self):
        """Test that functions compile with JIT"""
        jitted_D = jax.jit(D)
        jitted_f = jax.jit(f)
        
        z = 1.0
        omega_m = 0.3
        
        # Should produce same results
        assert jnp.isclose(jitted_D(z, omega_m), D(z, omega_m))
        assert jnp.isclose(jitted_f(z, omega_m), f(z, omega_m))
    
    def test_gradient_computation(self):
        """Test that gradients can be computed"""
        def loss_fn(params):
            z, omega_m = params
            return D(z, omega_m)**2
        
        grad_fn = jax.grad(loss_fn)
        
        params = jnp.array([1.0, 0.3])
        grads = grad_fn(params)
        
        # Gradients should be finite
        assert jnp.all(jnp.isfinite(grads))
    
    def test_vectorized_map(self):
        """Test vectorized operations with vmap"""
        z_values = jnp.array([0.0, 0.5, 1.0, 2.0])
        omega_m_values = jnp.array([0.2, 0.3, 0.4, 0.5])
        
        # Use vmap to vectorize over both arguments
        vmapped_D = jax.vmap(D)
        D_results = vmapped_D(z_values, omega_m_values)
        
        assert D_results.shape == (4,)
        assert jnp.all(jnp.isfinite(D_results))

class TestHypergeometricFunction:
    """Test the hypergeometric function implementation"""
    
    def test_hypergeometric_continuity(self):
        """Test continuity of hypergeometric function at x=0"""
        x_neg = -1e-6
        x_pos = 1e-6
        
        result_neg = _growth_2f1(x_neg)
        result_pos = _growth_2f1(x_pos)
        
        # Should be close at x=0
        assert jnp.isclose(result_neg, result_pos, rtol=1e-4)
    
    def test_hypergeometric_negative_domain(self):
        """Test hypergeometric function for negative arguments"""
        x_values = jnp.array([-10.0, -5.0, -1.0, -0.1])
        results = _growth_2f1(x_values)
        
        # Should be finite for all negative x
        assert jnp.all(jnp.isfinite(results))
        assert jnp.all(results > 0)  # Should be positive

# Pytest fixtures for common test data
@pytest.fixture
def standard_cosmology():
    """Standard cosmological parameters for testing"""
    return {"omega_m": 0.3, "z": 1.0}

@pytest.fixture
def redshift_array():
    """Array of redshift values for testing"""
    return jnp.logspace(-3, 1, 20)  # z from 0.001 to 10

@pytest.fixture
def omega_m_array():
    """Array of matter density values for testing"""
    return jnp.linspace(0.1, 0.9, 9)
