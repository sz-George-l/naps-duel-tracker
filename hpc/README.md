# HPC / SLURM job scripts

This directory contains SLURM sbatch scripts used to run different stages of
the behavioral analysis pipeline on an HPC cluster.

## Scripts

### Duel-tracking
Runs training or batch inference for the ST-GCN–based antennal dueling classifier.
Typically requires GPU resources.

### NAPS.h5 (CPU-dependent)
Runs NAPS identity assignment, SLEAP conversion, and H5 filtering.
Designed to run on CPU-only nodes.

### Prediction (GPU-dependent)
Applies trained models to pose-tracking `.h5` files to generate behavioral
predictions. Requires GPU resources.

## Notes
- These scripts are provided as examples and may require modification
  depending on cluster configuration.
- Paths, resource requests, and module loading commands are cluster-specific.
