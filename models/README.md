## Model overview

### duel-tracker_model
ST-GCN–based classifier for detecting antennal dueling (charging) behavior
from pairwise spatiotemporal pose trajectories.

### harpegnathos_saltator_200frames.centered_instance
Species-specific SLEAP pose estimation model trained on 200 manually
annotated frames using centered-instance mode. Outputs full-body keypoints
for multiple individuals.

### harpegnathos_saltator_200frames.centroid
Centroid-based SLEAP model trained on the same 200-frame dataset. Used for
coarse localization and identity association prior to full pose estimation.
