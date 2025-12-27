 # Installation

This pipeline is organized into three stages, each running in its own conda environment:

- **sleap**: pose prediction (GPU)
- **naps**: identity assignment + filtering (CPU)
- **duel-tracker**: interaction detection with an ST-GCN model (GPU)

The **SLEAP** and **NAPS** environments are upstream-maintained projects. We do not attempt to freeze or fully vendor their dependencies here. Instead, we provide lightweight conda environment wrappers in `envs/` that have been tested with this pipeline.

> GPU/CUDA setup varies by machine (local workstation vs HPC). Please follow the upstream installation guidance for your system when installing SLEAP and its GPU dependencies.

---

## Prerequisites

- **Conda** (Miniconda or Anaconda)
