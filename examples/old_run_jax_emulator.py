import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_fast_min_max=true '
    '--xla_gpu_strict_conv_algorithm_picker=false '
)

import jax
jax.config.update('jax_default_matmul_precision', 'high')
jax.config.update('jax_numpy_dtype_promotion', 'strict')
jax.config.update('jax_compilation_cache_dir', '.jax_cache')
jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)
jax.config.update('jax_persistent_cache_min_compile_time_secs', 0)

import time
import sys
import argparse
from glob import glob
from pathlib import Path
import numpy as np

from jax_nbody_emulator import create_emulator, SubboxConfig

# --- Validation functions ---
def validate_readable_file(p: Path):
    if not p.is_file():
        sys.exit(f'Input file path is not a readable file: {p}')
    try:
        with p.open('rb'):
            pass
    except Exception as e:
        sys.exit(f'Input file cannot be read: {p} ({e})')

def validate_writable_dir(p: Path):
    if not p.is_dir():
        sys.exit(f'Output directory path is not a directory: {p}')
    try:
        test = p / '.write_test'
        with test.open('w'):
            pass
        test.unlink()
    except Exception as e:
        sys.exit(f'Output directory is not writable: {p} ({e})')

def int_or_tuple(value: str):
    """Parse either a single int or a tuple of 3 ints."""
    parts = [int(x) for x in value.strip('()').split(',')]
    if len(parts) == 1:
        return (parts[0],) * 3
    if len(parts) == 3:
        return tuple(parts)
    raise argparse.ArgumentTypeError(f'Expected 1 or 3 values, got {len(parts)}')

def glob_str_r(value: str):
    paths = sorted([Path(p) for p in glob(value)])
    if not paths:
        raise argparse.ArgumentTypeError(f'No files match pattern: {value}')
    for p in paths:
        validate_readable_file(p)
    return paths

def glob_str_w(value: str):
    paths = sorted([Path(p) for p in glob(value)])
    if not paths:
        raise argparse.ArgumentTypeError(f'No files match pattern: {value}')
    for p in paths:
        validate_writable_dir(p)
    return paths

def set_precision(value: str):
    if value == 'f16':
        return jax.numpy.float16
    elif value == 'f32':
        return jax.numpy.float32
    raise argparse.ArgumentTypeError(f'--precision must \'f32\' or \'f16\'')

def validate_displacement_file(fn, in_shape):
    shape = np.load(fn, mmap_mode='r').shape
    if len(shape) != 4:
        sys.exit(f'in file {fn}: input array ndim {len(shape)} is not 4')
    if shape[0] != 3:
        sys.exit(f'in file {fn}: first dimension {shape[0]} is not 3')
    if in_shape is not None and shape != in_shape:
        sys.exit(f'in file {fn}: input array shape {shape} differs from shape in first file {in_shape}')
    return shape

def load_cosmology(fn):
    data = np.load(fn)
    Om, z = data[0], data[-1]
    if not 0.1 <= Om <= 0.5:
        sys.exit(f'in file {fn}: Om={Om:.4f} out of bounds [0.1, 0.5]')
    if not 0.0 <= z <= 3.0:
        sys.exit(f'in file {fn}: z={z:.4f} out of bounds [0.0, 3.0]')
    return Om, z

parser = argparse.ArgumentParser(
    description="Run JAX N-Body emulator with cosmology parameter files, displacement files, and output directories."
)
parser.add_argument('--cosmo_param_files', type=glob_str_r, required=True, help='Glob pattern for cosmology parameter files')
parser.add_argument('--displacement_files', type=glob_str_r, required=True, help='Glob pattern for displacement input files')
parser.add_argument('--output_dirs', type=glob_str_w, required=True, help='Glob pattern for output directories')
parser.add_argument('--ndiv', type=int_or_tuple, required=True, help='Number of divisions: single int (e.g., 4) or tuple (e.g., 2,4,4)')
parser.add_argument('--vel', action=argparse.BooleanOptionalAction, default=True, help='Compute velocities (default: True)')
parser.add_argument('--style', action=argparse.BooleanOptionalAction, default=True, help='Use one-the-fly style modulation, otherwise premodulate (default: True)')
parser.add_argument('--precision', type=set_precision, required=False, default='f32', help='Precision of model, either f16 (half) or f32 (full) (default: f32)')

def run_emulator(args):

    # --- Argument parsing ---
    if len(args.cosmo_param_files) != len(args.displacement_files) or len(args.cosmo_param_files) != len(args.output_dirs):
        parser.error('Number of cosmology files, displacement files, and output dirs must match')

    # --- Validate input and load cosmologies ---
    in_shape = None
    for dis_in_file in args.displacement_files:
        in_shape = validate_displacement_file(dis_in_file, in_shape)
    cosmo = [load_cosmology(fn) for fn in args.cosmo_param_files]
    Oms, zs = zip(*cosmo)

    # --- Setup emulator ---
    sb_config = SubboxConfig(
        size=in_shape[1:],
        ndiv=args.ndiv,
        dtype=args.precision
    )

    if args.style :
        emulator = create_emulator(
            premodulate=not args.style,
            compute_vel=args.vel,
            processor_config=sb_config,
        )

    # --- Process files ---
    for i, (dis_in_file, Om, z, out_dir) in enumerate(zip(args.displacement_files, Oms, zs, args.output_dirs)):
        start = time.time()


        if not args.style :
            emulator = create_emulator(
                premodulate=not args.style,
                compute_vel=args.vel,
                processor_config=sb_config,
                premodulate_Om = Om,
                premodulate_z = z,
            )

        dis_in = np.load(dis_in_file)
        out = emulator.process_box(dis_in, z=z, Om=Om)
        if args.vel:
            np.save(out_dir / 'emu_dis.npy', out[0])
            np.save(out_dir / 'emu_vel.npy', out[1])
        else:
            np.save(out_dir / 'emu_dis.npy', out)
        print(f'file {i+1}/{len(args.displacement_files)}, z={z:.4f} Om={Om:.4f}: {time.time() - start:.2f}s')

if __name__ == "__main__":
    args = parser.parse_args()
    run_emulator(args)
