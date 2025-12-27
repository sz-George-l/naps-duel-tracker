NAPS–Duel-Tracker

An end-to-end pipeline for identity-aware pose tracking and automated detection of social interactions in group-living animals.

This repository provides a three-stage, reproducible workflow for converting raw video recordings into quantitative social interaction events using pose estimation, identity assignment, and a spatiotemporal graph neural network (ST-GCN).

Overview

The pipeline is designed for high-throughput, long-term behavioral analysis in dense social settings (e.g. insect colonies), where identity preservation and interaction directionality are essential.

Pipeline summary:

MP4 video
  → SLEAP (pose prediction, GPU)
  → .slp
  → NAPS (identity assignment + filtering, CPU)
  → .h5
  → Duel-tracker (ST-GCN inference, GPU)
  → .csv (interaction events)


Each stage is isolated in its own conda environment to ensure reproducibility and to reflect differing hardware requirements.

Pipeline stages
Stage 1 — Pose prediction (SLEAP, GPU)

Input: raw .mp4 video

Output: pose predictions (.slp)

Environment: sleap

Hardware: GPU required

Stage 2 — Identity assignment & filtering (NAPS, CPU)

Input: SLEAP .slp

Output: identity-resolved and filtered pose data (.h5)

Environment: naps (plus sleap for format conversion)

Hardware: CPU only

Stage 3 — Interaction detection (Duel-tracker, GPU)

Input: filtered pose .h5

Output: interaction event tables (.csv)

Environment: duel-tracker

Hardware: GPU required

Repository structure
naps-duel-tracker/

├── envs/          # Conda environment definitions

├── hpc/           # SLURM batch scripts (GPU / CPU stages)

├── scripts/       # Helper scripts and utilities

├── src/           # Python modules (e.g. pose filtering)

├── models/        # Trained models (inference-ready)

├── data/

│   └── sample/    # Small public example dataset

├── docs/          # Detailed documentation

├── configs/       # YAML configuration files

└── README.md


Installation

This project requires conda.

Create the three environments:

conda env create -f envs/sleap.yml
conda env create -f envs/naps.yml
conda env create -f envs/duel-tracker.yml

Note: GPU/CUDA versions are system-dependent and are not hard-coded in the environment files. See docs/installation.md for recommended configurations.

ArUco dictionary requirement (NAPS)

NAPS relies on OpenCV’s ArUco module for identity assignment.
The official NAPS release (Python 3.7, legacy OpenCV) does not include a 3×3 dictionary, while our experiments use a custom 3×3 ArUco dictionary (512 tags).

To reproduce our results, one small modification is required.

Required change

Edit the following file inside your NAPS environment:

<conda_env>/lib/python3.7/site-packages/naps/aruco.py


Locate the line:

return cv2.aruco.Dictionary_get(ARUCO_DICT[tag_set])


and replace it with:

# Custom 3x3 ArUco dictionary (used in this project)
aruco_tag_dict = cv2.aruco.custom_dictionary(512, 3)

# IDs actually printed and used in experiments
desired_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

subset_dict = cv2.aruco.Dictionary_create(len(desired_id),
                                          aruco_tag_dict.markerSize)
subset_dict.bytesList = aruco_tag_dict.bytesList[desired_id]

return subset_dict

Note: This modification is required because OpenCV versions shipped with
Python 3.7 do not support extendDictionary().

Usage
HPC (SLURM)

Example execution order:

sbatch hpc/slurm_01_sleap_predict.sbatch
sbatch hpc/slurm_02_naps_and_filter.sbatch
sbatch hpc/slurm_03_duel_tracker.sbatch


Each stage checks for existing outputs and skips completed files, enabling safe re-runs.

Local testing

A minimal sample dataset is provided under data/sample/ to verify installation and pipeline logic without large data dependencies.

Data availability

Public:

All source code

Trained inference models

Conda environment files

SLURM scripts

A small representative sample dataset

Not included:

Full training datasets

Large-scale analyzed outputs

Long raw video recordings

Detailed descriptions of the training data and labeling procedures are provided in docs/training_data_description.md.

Design philosophy

Hybrid approach: deep learning for motion pattern recognition, deterministic rules for event logic and aggregation

Reproducibility first: explicit data handoffs and environment isolation

Scalability: designed for multi-day, multi-individual recordings

Interpretability: outputs are human-readable CSV tables suitable for downstream statistical and network analysis

Citation

If you use this pipeline, please cite:

Liu, S. Z. G., et al. (2025).
Internal-state asymmetry shapes social behaviors to determine caste fate.

(Full citation details in CITATION.cff.)

License

This project is released under an open-source license. See LICENSE for details.

Contact

Questions, issues, and contributions are welcome via GitHub Issues.
