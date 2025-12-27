## Script overview

### duel-tracker_v3
Training script for the ST-GCN–based antennal dueling classifier.
Takes filtered SLEAP pose `.h5` files and manually annotated dueling events
as input and outputs a trained `duel-tracker_model`.

### duel_tracker_prediction_v3
Inference script that applies a trained dueling classifier to new pose
trajectories and outputs per-window dueling probabilities.

### h5_filter
Utility script for filtering and cleaning pose-tracking `.h5` files prior
to training or inference.

## Pipeline overview

1. Pose estimation using SLEAP
2. Filtering of pose `.h5` files (`h5_filter`)
3. Training of dueling classifier (`duel-tracker_v3`)
4. Inference on new data (`duel_tracker_prediction_v3`)
