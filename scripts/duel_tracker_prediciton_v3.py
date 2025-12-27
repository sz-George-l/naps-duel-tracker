# predict_single_h5.py
import os
import json
import torch
import h5py
import pandas as pd
import numpy as np
from duel_tracker_v3 import (
    STGCNModel, extract_and_predict, cluster_by_peaks
)

def predict_single_file(model_dir, h5_file_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # === Load model configuration
    config_path = os.path.join(model_dir, "model_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    window = config.get("window_size", 60)
    stride = config.get("stride", 15)
    in_channels = config.get("in_channels", 6)
    hidden = config.get("hidden", 32)
    node_dropout_rate = config.get("node_dropout_rate", 0.2)
    cluster_distance = config.get("cluster_distance", 3)
    cluster_height = config.get("height", 0.1)
    best_cutoff = config.get("best_cutoff", 0.5)
    frame_limit = 108000

    print(f"🔍 Using model: {model_dir}")
    print(f"🔎 Applying cutoff threshold: {best_cutoff:.4f}")

    # === Load model
    A = np.load(os.path.join(model_dir, "adj_matrix.npy"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = STGCNModel(
        A=A,
        in_channels=in_channels,
        hidden=hidden,
        time_len=window,
        num_nodes=A.shape[0],
        node_dropout_rate=node_dropout_rate
    ).to(device)

    model_path = os.path.join(model_dir, "stgcn_duel_model.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # === Predict
    df, _ = extract_and_predict(
        h5_file=h5_file_path,
        label_file=None,
        model=model,
        device=device,
        window=window,
        stride=stride,
        frame_limit=frame_limit,
        augment=False
    )

    if df is None or df.empty:
        print("⚠️ No valid predictions generated.")
        return

    # === Cluster and apply cutoff
    df = cluster_by_peaks(df, distance=cluster_distance, height=cluster_height)
    df = df[df["Probability"] >= best_cutoff].copy()

    if df.empty:
        print("⚠️ No predictions passed the cutoff after clustering.")
        return

    # === Format output
    df_out = df[["Track", "Target", "DominantNode", "Time", "Frame", "Probability"]].copy()
    df_out.columns = ["track", "target", "target_node", "time", "frame", "probability"]

    # === Save CSV
    basename = os.path.splitext(os.path.basename(h5_file_path))[0]
    out_path = os.path.join(output_dir, f"{basename}_predictions.csv")
    df_out.to_csv(out_path, index=False)
    print(f"✅ Predictions saved to: {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run ST-GCN duel prediction on a single .h5 file.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to trained model directory")
    parser.add_argument("--h5_file", type=str, required=True, help="Path to a single .h5 tracking file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output prediction CSV")
    args = parser.parse_args()

    predict_single_file(args.model_dir, args.h5_file, args.output_dir)
