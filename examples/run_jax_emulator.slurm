#!/bin/bash -l
#===============================================================================
# SLURM Job Script: JAX N-body Emulator Batch Processing
#===============================================================================
#
# This script runs the JAX N-body emulator on multiple simulation outputs
# using a single GPU node. It processes displacement fields and outputs
# emulated displacement and velocity fields.
#
# Usage:
#   sbatch run_jax_emulator.slurm
#
# Before running:
#   1. Update the paths below to match your directory structure
#   2. Adjust --time based on number of files and settings (see estimates below)
#   3. Verify your conda/mamba environment name
#
# Time estimates for 512³ volumes on A100 (40GB):
#   FP16 + velocity:     ~26s per file
#   FP16 + no velocity:  ~11s per file
#   FP32 + velocity:     ~45s per file
#   FP32 + no velocity:  ~16s per file
#
#===============================================================================

#-------------------------------------------------------------------------------
# SLURM Configuration
#-------------------------------------------------------------------------------

#SBATCH --job-name=jaxemu
#SBATCH --output=%x_out.txt

# Node configuration
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18

# GPU configuration (adjust GPU type as needed: a100, v100, etc.)
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1

# Resource limits
#SBATCH --mem=125000M
#SBATCH --time=00-08:00:00

#-------------------------------------------------------------------------------
# Environment Setup
#-------------------------------------------------------------------------------

# Activate conda/mamba environment with JAX installed
# Uncomment and modify the appropriate line for your setup:
# source activate jax
# conda activate jax
mamba activate jax

#-------------------------------------------------------------------------------
# Input/Output Configuration
#-------------------------------------------------------------------------------

# Base paths - MODIFY THESE FOR YOUR SETUP
cosmo_param_base="/path/to/cosmology/params"
dis_in_base="/path/to/input/displacements"
out_base="/path/to/output"

# Glob patterns for file matching
# The patterns must align: the Nth cosmology file corresponds to the Nth
# displacement file and Nth output directory.
#
# Example directory structure:
#   /path/to/cosmology/params/Fiducial/snap6/params.npy
#   /path/to/input/displacements/Fiducial/snap6/dis.npy
#   /path/to/output/Fiducial/snap6/
#
# Pattern breakdown:
#   First  *: cosmology label (e.g., Fiducial, Om_high, Om_low)
#   Second *: snapshot/redshift label (e.g., snap0, snap6, z0.5)

cosmo_param_files="${cosmo_param_base}/*/*/params.npy"
dis_in_files="${dis_in_base}/*/*/dis.npy"
out_dirs="${out_base}/*/*"

#-------------------------------------------------------------------------------
# Emulator Configuration
#-------------------------------------------------------------------------------

# Subbox divisions along each spatial dimension
# More divisions = smaller subboxes = less GPU memory, but more overhead
# Recommended for 512³ volumes:
#   FP16: (4,2,2) -> 16 subboxes of 128x256x256
#   FP32: (4,4,4) -> 64 subboxes of 128x128x128 (with velocity)
#   FP32: (4,4,2) -> 32 subboxes of 128x128x256 (without velocity)
ndiv="4,2,2"

# Floating point precision: f16 (faster, less memory) or f32 (more accurate)
precision="f16"

# Output precision: f16 (smaller files, less memory) or f32 (more accurate)
# Use f16 for large volumes (e.g., 1536³) to avoid OOM during output assembly
output_precision="f16"

# Velocity computation: include --vel to compute velocities, omit to skip
# Computing velocities approximately doubles the processing time
compute_vel="--vel"
# compute_vel=""  # Uncomment to disable velocity computation

# Style modulation: include --style for flexible cosmology (single emulator),
# omit for premodulated mode (new emulator per cosmology, slightly faster)
style_mode="--style"
# style_mode=""  # Uncomment to use premodulated mode

# Quiet mode: include --quiet to suppress progress bars
# quiet_mode="--quiet"
quiet_mode=""

#-------------------------------------------------------------------------------
# Run Emulator
#-------------------------------------------------------------------------------

echo "============================================================"
echo "JAX N-body Emulator Batch Job"
echo "============================================================"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""
echo "Configuration:"
echo "  Precision: ${precision}"
echo "  Output precision: ${output_precision}"
echo "  Divisions: ${ndiv}"
echo "  Velocity:  ${compute_vel:-disabled}"
echo "  Style:     ${style_mode:-premodulated}"
echo ""
echo "Input patterns:"
echo "  Cosmo:     ${cosmo_param_files}"
echo "  Displace:  ${dis_in_files}"
echo "  Output:    ${out_dirs}"
echo "============================================================"
echo ""

srun --cpu-bind=cores python -u examples/run_jax_emulator.py \
    --cosmo_param_files "${cosmo_param_files}" \
    --displacement_files "${dis_in_files}" \
    --output_dirs "${out_dirs}" \
    --ndiv "${ndiv}" \
    --precision "${precision}" \
    --output-precision "${output_precision}" \
    ${style_mode} \
    ${compute_vel} \
    ${quiet_mode}

echo ""
echo "============================================================"
echo "End time: $(date)"
echo "============================================================"

