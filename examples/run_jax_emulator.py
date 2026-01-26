#!/usr/bin/env python
"""
Batch processing script for the JAX N-body emulator.

This script processes multiple displacement fields with their corresponding
cosmological parameters, saving the output displacement and velocity fields.

Example usage:
    python run_jax_emulator.py \
        --cosmo_param_files '/path/to/sims/*/params.npy' \
        --displacement_files '/path/to/sims/*/dis.npy' \
        --output_dirs '/path/to/sims/*/' \
        --ndiv 4,2,2 \
        --precision f16 \
        --vel

Note: Progress bars are printed to stderr by default. To redirect to stdout:
    python run_jax_emulator.py ... 2>&1

Input file formats:
    - Cosmology files: numpy array with shape (6,) containing:
        [Omega_m, Omega_b, h, n_s, sigma_8, redshift]
    - Displacement files: numpy array with shape (3, N, N, N)
        representing the z=0 linear (ZA) displacement field

Output files:
    - emu_dis.npy: Emulated displacement field (3, N, N, N)
    - emu_vel.npy: Emulated velocity field (3, N, N, N) [if --vel is set]

Copyright (C) 2025 Drew Jamieson
Licensed under GNU GPL v3.0 - see LICENSE file for details.

Author: Drew Jamieson <drew.s.jamieson@gmail.com>
"""

import os

# Suppress TensorFlow logging and configure XLA for faster convolutions
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_fast_min_max=true '
    '--xla_gpu_strict_conv_algorithm_picker=false '
)

import jax
import jax.numpy as jnp

# JAX configuration for performance and caching
jax.config.update('jax_default_matmul_precision', 'high')
jax.config.update('jax_numpy_dtype_promotion', 'strict')
jax.config.update('jax_compilation_cache_dir', '.jax_cache')
jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)
jax.config.update('jax_persistent_cache_min_compile_time_secs', 0)

import sys
import time
import argparse
from glob import glob
from pathlib import Path

import numpy as np

from jax_nbody_emulator import create_emulator, SubboxConfig


# =============================================================================
# Validation utilities
# =============================================================================

def validate_readable_file(path: Path) -> None:
    """Check that a file exists and is readable."""
    if not path.is_file():
        sys.exit(f'Input file path is not a readable file: {path}')
    try:
        with path.open('rb'):
            pass
    except Exception as e:
        sys.exit(f'Input file cannot be read: {path} ({e})')


def validate_writable_dir(path: Path) -> None:
    """Check that a directory exists and is writable."""
    if not path.is_dir():
        sys.exit(f'Output directory path is not a directory: {path}')
    try:
        test = path / '.write_test'
        with test.open('w'):
            pass
        test.unlink()
    except Exception as e:
        sys.exit(f'Output directory is not writable: {path} ({e})')


def validate_displacement_file(filepath: Path, expected_shape: tuple | None) -> tuple:
    """
    Validate displacement file format and shape consistency.
    
    Args:
        filepath: Path to displacement .npy file
        expected_shape: Expected shape from previous files (None for first file)
    
    Returns:
        Shape of the displacement array (3, N, N, N)
    """
    shape = np.load(filepath, mmap_mode='r').shape
    
    if len(shape) != 4:
        sys.exit(f'in file {filepath}: input array ndim {len(shape)} is not 4')
    if shape[0] != 3:
        sys.exit(f'in file {filepath}: first dimension {shape[0]} is not 3 (expected 3 displacement components)')
    if expected_shape is not None and shape != expected_shape:
        sys.exit(f'in file {filepath}: input array shape {shape} differs from first file shape {expected_shape}')
    
    return shape


def load_cosmology(filepath: Path) -> tuple[float, float]:
    """
    Load cosmological parameters from file.
    
    Expected file format: numpy array with shape (6,) containing:
        [Omega_m, Omega_b, h, n_s, sigma_8, redshift]
    
    Args:
        filepath: Path to cosmology parameters .npy file
    
    Returns:
        Tuple of (Omega_m, redshift)
    """
    data = np.load(filepath)
    Om, z = float(data[0]), float(data[-1])
    
    # Validate parameter ranges
    if not 0.1 <= Om <= 0.5:
        sys.exit(f'in file {filepath}: Om={Om:.4f} out of valid range [0.1, 0.5]')
    if not 0.0 <= z <= 3.0:
        sys.exit(f'in file {filepath}: z={z:.4f} out of valid range [0.0, 3.0]')
    
    return Om, z


# =============================================================================
# Argument parsing utilities
# =============================================================================

def int_or_tuple(value: str) -> tuple[int, int, int]:
    """
    Parse subbox divisions as single int or tuple of 3 ints.
    
    Examples:
        "4" -> (4, 4, 4)
        "2,4,4" -> (2, 4, 4)
        "(2, 4, 4)" -> (2, 4, 4)
    """
    parts = [int(x) for x in value.strip('()').split(',')]
    if len(parts) == 1:
        return (parts[0],) * 3
    if len(parts) == 3:
        return tuple(parts)
    raise argparse.ArgumentTypeError(f'Expected 1 or 3 values, got {len(parts)}')


def glob_readable(pattern: str) -> list[Path]:
    """Parse glob pattern and validate all matched files are readable."""
    paths = sorted([Path(p) for p in glob(pattern)])
    if not paths:
        raise argparse.ArgumentTypeError(f'No files match pattern: {pattern}')
    for p in paths:
        validate_readable_file(p)
    return paths


def glob_writable(pattern: str) -> list[Path]:
    """Parse glob pattern and validate all matched directories are writable."""
    paths = sorted([Path(p) for p in glob(pattern)])
    if not paths:
        raise argparse.ArgumentTypeError(f'No directories match pattern: {pattern}')
    for p in paths:
        validate_writable_dir(p)
    return paths


def parse_precision(value: str):
    """Parse precision string to dtype (works for both JAX and NumPy)."""
    if value == 'f16':
        return np.float16
    elif value == 'f32':
        return np.float32
    raise argparse.ArgumentTypeError(f'precision must be \'f32\' or \'f16\', got \'{value}\'')


# =============================================================================
# Main processing
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser with all options."""
    parser = argparse.ArgumentParser(
        description='Batch process displacement fields with the JAX N-body emulator.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all simulations in a directory structure
  python run_jax_emulator.py \\
      --cosmo_param_files '/data/sims/*/snap6/params.npy' \\
      --displacement_files '/data/sims/*/snap6/dis.npy' \\
      --output_dirs '/data/sims/*/snap6/' \\
      --ndiv 4,2,2 --precision f16 --vel

  # Process without velocities, using premodulation
  python run_jax_emulator.py \\
      --cosmo_param_files 'sim*/params.npy' \\
      --displacement_files 'sim*/dis.npy' \\
      --output_dirs 'sim*/' \\
      --ndiv 4 --no-vel --no-style
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--cosmo_param_files', type=glob_readable, required=True,
        help='Glob pattern for cosmology parameter files (numpy arrays with [Om, Ob, h, ns, s8, z])'
    )
    parser.add_argument(
        '--displacement_files', type=glob_readable, required=True,
        help='Glob pattern for input displacement files (numpy arrays with shape [3, N, N, N])'
    )
    parser.add_argument(
        '--output_dirs', type=glob_writable, required=True,
        help='Glob pattern for output directories'
    )
    parser.add_argument(
        '--ndiv', type=int_or_tuple, required=True,
        help='Number of subbox divisions: single int (e.g., 4) or tuple (e.g., 2,4,4)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--vel', action=argparse.BooleanOptionalAction, default=True,
        help='Compute velocity field in addition to displacement (default: True)'
    )
    parser.add_argument(
        '--style', action=argparse.BooleanOptionalAction, default=True,
        help='Use style modulation for flexible cosmology; if False, premodulate parameters '
             'for each cosmology (default: True)'
    )
    parser.add_argument(
        '--precision', type=parse_precision, default=np.float32,
        help='Model precision: f16 (half) or f32 (full) (default: f32)'
    )
    parser.add_argument(
        '--output-precision', type=parse_precision, default=np.float16,
        dest='output_precision',
        help='Output file precision: f16 (half) or f32 (full) (default: f16). '
             'Use f16 for large volumes to reduce memory usage.'
    )
    parser.add_argument(
        '--quiet', '-q', action='store_true',
        help='Suppress progress bars (useful for batch jobs)'
    )
    
    return parser


def run_emulator(args: argparse.Namespace) -> None:
    """
    Main processing loop.
    
    Args:
        args: Parsed command-line arguments
    """
    # Validate that all glob patterns matched the same number of files
    n_cosmo = len(args.cosmo_param_files)
    n_disp = len(args.displacement_files)
    n_out = len(args.output_dirs)
    
    if not (n_cosmo == n_disp == n_out):
        sys.exit(
            f'Number of files must match:\n'
            f'  cosmo_param_files: {n_cosmo}\n'
            f'  displacement_files: {n_disp}\n'
            f'  output_dirs: {n_out}'
        )
    
    print(f'Processing {n_cosmo} simulation(s)')
    print(f'  Precision: {args.precision}')
    print(f'  Output precision: {args.output_precision}')
    print(f'  Compute velocity: {args.vel}')
    print(f'  Style modulation: {args.style}')
    print(f'  Subbox divisions: {args.ndiv}')
    print()
    
    # Validate all displacement files and get consistent shape
    input_shape = None
    for dis_file in args.displacement_files:
        input_shape = validate_displacement_file(dis_file, input_shape)
    
    box_size = input_shape[1:]  # (N, N, N)
    print(f'  Box size: {box_size}')
    
    # Load all cosmological parameters
    cosmologies = [load_cosmology(fn) for fn in args.cosmo_param_files]
    
    # Configure subbox processing
    config = SubboxConfig(
        size=box_size,
        ndiv=args.ndiv,
        dtype=args.precision,
        output_dtype=args.output_precision,
    )
    
    # Create emulator (single instance for style mode, per-cosmology for premodulated)
    emulator = None
    if args.style:
        emulator = create_emulator(
            premodulate=False,
            compute_vel=args.vel,
            processor_config=config,
        )
    
    # Process each simulation
    for i, (dis_file, cosmo_file, out_dir) in enumerate(
        zip(args.displacement_files, args.cosmo_param_files, args.output_dirs)
    ):
        Om, z = cosmologies[i]
        
        # For premodulated mode, create new emulator for each cosmology
        if not args.style:
            emulator = create_emulator(
                premodulate=True,
                premodulate_z=z,
                premodulate_Om=Om,
                compute_vel=args.vel,
                processor_config=config,
            )
        
        # Load input displacement field
        dis_in = np.load(dis_file)
        
        # Run emulator
        start = time.time()
        output = emulator.process_box(dis_in, z=z, Om=Om, show_progress=not args.quiet)
        elapsed = time.time() - start
        
        # Save outputs
        if args.vel:
            dis_out, vel_out = output
            np.save(out_dir / 'emu_dis.npy', dis_out)
            np.save(out_dir / 'emu_vel.npy', vel_out)
        else:
            np.save(out_dir / 'emu_dis.npy', output)
        
        print(f'[{i+1}/{n_cosmo}] z={z:.4f}, Om={Om:.4f}: {elapsed:.2f}s -> {out_dir}')
    
    print('\nDone!')


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    run_emulator(args)
