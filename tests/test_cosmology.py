"""
Tests for cosmology.py module.

Copyright (C) 2025 Drew Jamieson
Licensed under GNU GPL v3.0 - see LICENSE file for details.
"""
import pytest
import jax
import jax.numpy as jnp
from jax_nbody_emulator.cosmology import (
    D, H, f, vel_norm, acc_norm, dlogH_dloga, _growth_2f1
)


class TestBasicCosmology:
    """Test basic cosmological functions"""
    
    def test_growth_factor_at_z_zero(self):
        """Growth factor should be 1 at z=0"""
        z = jnp.array([0.0])
        assert jnp.isclose(D(z, jnp.array([0.3])), 1.0, rtol=1e-6)
        assert jnp.isclose(D(z, jnp.array([0.1])), 1.0, rtol=1e-6)
        assert jnp.isclose(D(z, jnp.array([0.5])), 1.0, rtol=1e-6)
    
    def test_growth_factor_decreases_with_redshift(self):
        """Growth factor should decrease with increasing redshift"""
        Om = jnp.array([0.3, 0.3, 0.3, 0.3, 0.3])
        z_values = jnp.array([0.0, 0.5, 1.0, 2.0, 3.0])
        D_values = D(z_values, Om)
        
        # Should be monotonically decreasing
        assert jnp.all(jnp.diff(D_values) < 0)
    
    def test_hubble_parameter_at_z_zero(self):
        """Hubble parameter should equal 100 at z=0 (in units of h km/s/Mpc)"""
        z = jnp.array([0.0])
        assert jnp.isclose(H(z, jnp.array([0.3])), 100.0, rtol=1e-6)
        assert jnp.isclose(H(z, jnp.array([0.1])), 100.0, rtol=1e-6)
    
    def test_hubble_parameter_increases_with_redshift(self):
        """Hubble parameter should increase with redshift"""
        Om = jnp.array([0.3, 0.3, 0.3, 0.3, 0.3])
        z_values = jnp.array([0.0, 0.5, 1.0, 2.0, 3.0])
        H_values = H(z_values, Om)
        
        # Should be monotonically increasing
        assert jnp.all(jnp.diff(H_values) > 0)
    
    def test_growth_rate_positive(self):
        """Growth rate f should be positive"""
        Om = jnp.array([0.3, 0.3, 0.3, 0.3])
        z_values = jnp.array([0.0, 0.5, 1.0, 2.0])
        f_values = f(z_values, Om)
        
        assert jnp.all(f_values > 0)
        assert jnp.all(f_values < 2.0)  # Reasonable upper bound


class TestCosmologyArrays:
    """Test that functions work with arrays"""
    
    def test_vectorized_operations(self):
        """Test that functions work with array inputs"""
        z_array = jnp.array([0.0, 0.5, 1.0, 2.0])
        Om_array = jnp.array([0.3, 0.3, 0.3, 0.3])
        
        # Test with array inputs
        D_values = D(z_array, Om_array)
        assert D_values.shape == z_array.shape
        
        H_values = H(z_array, Om_array)
        assert H_values.shape == z_array.shape
        
        f_values = f(z_array, Om_array)
        assert f_values.shape == z_array.shape
    
    def test_different_omega_m_values(self):
        """Test with different Omega_m values"""
        z_array = jnp.array([1.0, 1.0, 1.0, 1.0])
        Om_array = jnp.array([0.2, 0.3, 0.4, 0.5])
        
        D_values = D(z_array, Om_array)
        assert D_values.shape == Om_array.shape
        
        # Higher omega_m should give lower growth at fixed z
        assert jnp.all(jnp.diff(D_values) < 0)


class TestCosmologyDerivatives:
    """Test derivative functions"""
    
    def test_growth_rate_finite_difference(self):
        """Test growth rate against finite difference"""
        z = jnp.array([1.0])
        Om = jnp.array([0.3])
        dz = 1e-4
        
        # Finite difference approximation of f = d log D / d log a
        z_plus = jnp.array([z[0] + dz])
        z_minus = jnp.array([z[0] - dz])
        
        dlogD_dz_fd = (jnp.log(D(z_plus, Om)) - jnp.log(D(z_minus, Om))) / (2 * dz)
        f_fd = -dlogD_dz_fd[0] * (1 + z[0])
        
        # Automatic differentiation result
        f_ad = f(z, Om)[0]
        
        assert jnp.isclose(f_ad, f_fd, rtol=1e-3)
    
    def test_hubble_derivative(self):
        """Test Hubble derivative function"""
        z = jnp.array([1.0])
        Om = jnp.array([0.3])
        
        # Should be finite
        dlogH_dloga_val = dlogH_dloga(z, Om)[0]
        assert jnp.isfinite(dlogH_dloga_val)
        
        # Should be negative (H decreases as a increases)
        assert dlogH_dloga_val < 0


class TestCosmologyPhysics:
    """Test physical consistency"""
    
    def test_einstein_de_sitter_limit(self):
        """Test Einstein-de Sitter limit (Om=1)"""
        Om = jnp.array([0.99999])  # Close to 1
        z = jnp.array([1.0])
        
        # In EdS: D(z) = 1/(1+z), f = 1
        D_eds = D(z, Om)[0]
        f_eds = f(z, Om)[0]
        
        expected_D = 1.0 / (1 + z[0])
        expected_f = 1.0
        
        assert jnp.isclose(D_eds, expected_D, rtol=1e-3)
        assert jnp.isclose(f_eds, expected_f, rtol=1e-2)
    
    def test_velocity_and_acceleration_units(self):
        """Test velocity and acceleration normalization factors"""
        z = jnp.array([1.0])
        Om = jnp.array([0.3])
        
        vel = vel_norm(z, Om)[0]
        acc = acc_norm(z, Om)[0]
        
        # Should be positive and finite
        assert vel > 0 and jnp.isfinite(vel)
        assert jnp.isfinite(acc)  # Can be positive or negative
        
        # Velocity should have units of km/s (order of magnitude check)
        assert 10 < vel < 1000  # Reasonable range
    
    def test_omega_m_dependence(self):
        """Test dependence on matter density parameter"""
        z = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
        Om_values = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
        
        D_values = D(z, Om_values)
        f_values = f(z, Om_values)
        
        # Higher omega_m should give lower growth at fixed z if D(z=0)=1
        assert jnp.all(jnp.diff(D_values) < 0)
        # Growth rate should increase with omega_m
        assert jnp.all(jnp.diff(f_values) > 0)


class TestCosmologyEdgeCases:
    """Test edge cases and numerical stability"""
    
    def test_high_redshift_behavior(self):
        """Test behavior at high redshift"""
        Om = jnp.array([0.3, 0.3, 0.3])
        z_high = jnp.array([5.0, 10.0, 20.0])
        
        D_high = D(z_high, Om)
        f_high = f(z_high, Om)
        
        # Should be finite
        assert jnp.all(jnp.isfinite(D_high))
        assert jnp.all(jnp.isfinite(f_high))
        
        # Should be small at high z
        assert jnp.all(D_high < 0.25)
        
        # f should approach Om(z)^0.55 at high z
        # where Om(z) = Om(1+z)^3 / [Om(1+z)^3 + OL]
        Om_val = Om[0]
        OL = 1 - Om_val
        Om_z = Om_val * (1 + z_high)**3 / (Om_val * (1 + z_high)**3 + OL)
        expected_f = Om_z ** 0.55
        assert jnp.all(jnp.isclose(f_high, expected_f, rtol=0.01))
    
    def test_very_small_omega_m(self):
        """Test behavior with very small omega_m"""
        Om = jnp.array([1e-6])
        z = jnp.array([1.0])
        
        # Should still be finite
        D_small = D(z, Om)[0]
        f_small = f(z, Om)[0]
        
        assert jnp.isfinite(D_small)
        assert jnp.isfinite(f_small)
    
    def test_zero_redshift(self):
        """Test behavior at z=0"""
        Om = jnp.array([0.3])
        z = jnp.array([0.0])
        
        D_zero = D(z, Om)[0]
        assert jnp.isclose(D_zero, 1.0, rtol=1e-6)


class TestJAXCompatibility:
    """Test JAX-specific functionality"""
    
    def test_jit_compilation(self):
        """Test that functions compile with JIT"""
        # Functions are already JIT compiled, test they work
        z = jnp.array([1.0])
        Om = jnp.array([0.3])
        
        D_val = D(z, Om)[0]
        H_val = H(z, Om)[0]
        f_val = f(z, Om)[0]
        
        assert jnp.isfinite(D_val)
        assert jnp.isfinite(H_val)
        assert jnp.isfinite(f_val)
    
    def test_gradient_computation(self):
        """Test that gradients can be computed"""
        def loss_fn(z_val):
            z = jnp.array([z_val])
            Om = jnp.array([0.3])
            return D(z, Om)[0]**2
        
        grad_fn = jax.grad(loss_fn)
        grad = grad_fn(1.0)
        
        # Gradient should be finite
        assert jnp.isfinite(grad)
    
    def test_batch_processing(self):
        """Test batch processing of multiple z and Om values"""
        batch_size = 10
        z_batch = jnp.linspace(0.0, 2.0, batch_size)
        Om_batch = jnp.full(batch_size, 0.3)
        
        D_batch = D(z_batch, Om_batch)
        vel_batch = vel_norm(z_batch, Om_batch)
        
        assert D_batch.shape == (batch_size,)
        assert vel_batch.shape == (batch_size,)
        assert jnp.all(jnp.isfinite(D_batch))
        assert jnp.all(jnp.isfinite(vel_batch))


class TestHypergeometricFunction:
    """Test the hypergeometric function implementation"""
    
    def test_hypergeometric_continuity(self):
        """Test continuity of hypergeometric function at x=0"""
        x_neg = jnp.array(-1e-6)
        x_pos = jnp.array(1e-6)
        
        result_neg = _growth_2f1(x_neg)
        result_pos = _growth_2f1(x_pos)
        
        # Should be close at x=0
        assert jnp.isclose(result_neg, result_pos, rtol=1e-3)
    
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
    return {"Om": jnp.array([0.3]), "z": jnp.array([1.0])}


@pytest.fixture
def redshift_array():
    """Array of redshift values for testing"""
    return jnp.logspace(-3, 1, 20)  # z from 0.001 to 10


@pytest.fixture
def omega_m_array():
    """Array of matter density values for testing"""
    return jnp.linspace(0.1, 0.9, 9)
