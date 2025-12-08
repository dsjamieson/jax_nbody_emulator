"""
Cosmological functions for flat LambdaCDM universe.

This module provides JAX-accelerated functions for computing various
cosmological quantities in a flat LambdaCDM universe model.

Copyright (C) 2025 Drew Jamieson
Licensed under GNU GPL v3.0 - see LICENSE file for details.

Author: Drew Jamieson <drew.s.jamieson@gmail.com>
"""

import jax
import jax.numpy as jnp
from jax import custom_jvp
from jax.scipy.special import hyp2f1

# Parameters for hyp2f1(1, 1/3, 11/6, x)
_a2f1 = jnp.array(1.0, dtype=jnp.float32)
_b2f1 = jnp.array(1.0/3.0, dtype=jnp.float32)
_c2f1 = jnp.array(11.0/6.0, dtype=jnp.float32)

@jax.jit
def _growth_2f1(x):
    """Compute hyp2f1(1, 1/3, 11/6, x) for x < 0"""
    def compute_neg(x):
        z = x / (x - 1)
        return jnp.power(1 - x, -_a2f1) * hyp2f1(_a2f1, _c2f1 - _b2f1, _c2f1, z)
    def compute_pos(x):
        return hyp2f1(_a2f1, _b2f1, _c2f1, x)
    return jnp.where(x < 0, compute_neg(x), compute_pos(x))

@jax.jit
def D(z, Om):
    """Linear growth function for flat LambdaCDM, normalized to 1 at redshift zero"""
    a = 1 / (1 + z)
    OL = 1 - Om
    aa3 = -OL * a**3 / Om
    aa30 = -OL / Om
    return a * _growth_2f1(aa3) / _growth_2f1(aa30)

@jax.jit
def H(z, Om):
    """Hubble parameter in [h km/s/Mpc] for flat LambdaCDM"""
    OL = 1 - Om
    return 100 * jnp.sqrt(Om * (1 + z)**3 + OL)

@jax.jit
def _log_D(z, Om):
    """Log of linear growth function"""
    return jnp.log(D(z, Om))

@jax.jit
def _log_H(z, Om):
    """Log of Hubble parameter"""
    return jnp.log(H(z, Om))

# Use forward-mode differentiation (JVP) instead of reverse-mode (grad)
def _dlogD_dz_scalar(z_scalar, Om_scalar):
    """Scalar derivative of log D with respect to z using forward-mode AD."""
    primals = (z_scalar,)
    tangents = (jnp.ones_like(z_scalar),)
    _, deriv = jax.jvp(lambda z: _log_D(z, Om_scalar), primals, tangents)
    return deriv

def _dlogH_dz_scalar(z_scalar, Om_scalar):
    """Scalar derivative of log H with respect to z using forward-mode AD."""
    primals = (z_scalar,)
    tangents = (jnp.ones_like(z_scalar),)
    _, deriv = jax.jvp(lambda z: _log_H(z, Om_scalar), primals, tangents)
    return deriv

# Vectorized derivatives
_dlogD_dz = jax.jit(jax.vmap(_dlogD_dz_scalar, in_axes=(0, 0)))
_dlogH_dz = jax.jit(jax.vmap(_dlogH_dz_scalar, in_axes=(0, 0)))

@jax.jit
def f(z, Om):
    """
    Linear growth rate for flat LambdaCDM
    f = d log D / d log a
    
    Args:
        z: Redshift(s) of shape (B,)
        Om: Omega_matter(s) of shape (B,)
    
    Returns:
        Growth rate(s) of shape (B,)
    """
    dlogD_dz_val = _dlogD_dz(z, Om)
    return -dlogD_dz_val * (1 + z)

@jax.jit
def dlogH_dloga(z, Om):
    """
    Log-log derivative of Hubble w.r.t scale factor
    
    Args:
        z: Redshift(s) of shape (B,)
        Om: Omega_matter(s) of shape (B,)
    
    Returns:
        Derivative(s) of shape (B,)
    """
    dlogH_dz_val = _dlogH_dz(z, Om)
    return -dlogH_dz_val * (1 + z)

@jax.jit
def vel_norm(z, Om):
    """
    Velocity normalization factor [km/s]
    
    Args:
        z: Redshift(s) of shape (B,)
        Om: Omega_matter(s) of shape (B,)
    
    Returns:
        Normalization factor(s) of shape (B,)
    """
    return D(z, Om) * f(z, Om) * H(z, Om) / (1 + z)

@jax.jit
def acc_norm(z, Om):
    """
    Acceleration normalization factor [km/s^2]
    
    Args:
        z: Redshift(s) of shape (B,)
        Om: Omega_matter(s) of shape (B,)
    
    Returns:
        Normalization factor(s) of shape (B,)
    """
    return D(z, Om) * f(z, Om) * H(z, Om)**2 * dlogH_dloga(z, Om) / (1 + z)
