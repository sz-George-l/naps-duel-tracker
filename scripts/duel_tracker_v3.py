import os
import h5py
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import re
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score
from scipy.signal import find_peaks

WINDOWSIZE = 60
STRIDE = 5
FRAME_LIMIT = 18000

# === Helper Functions ===
def time_to_seconds(t):
    try:
        m, s = str(t).strip().split(":")
        return int(m) * 60 + int(s)
    except:
        return None
        
def load_skeleton_from_h5(h5_path):
    with h5py.File(h5_path, "r") as f:
        coords = f["tracks"][:FRAME_LIMIT].T  # [T, J, 2, N] up to FRAME_LIMIT
        names = [n.decode().strip() for n in f["track_names"][:]]
    return {name: coords[:, :, :, i] for i, name in enumerate(names)}

def parse_duel_labels(label_path, fps=30):
    df = pd.read_csv(label_path, header=None)
    labels = {}
    for _, row in df.iterrows():
        track = str(row[0]).strip()
        times = [int(m)*60 + int(s) for val in row[1:] if pd.notna(val)
                 for m, s in re.findall(r"(\d+):(\d+)", str(val))]
        labels[track] = set([t * fps for t in times])
    return labels

def build_adjacency_matrix_with_target():
    """
    Builds a 12x12 adjacency matrix connecting 11 ego joints and 1 target tag node.
    Returns:
        A (np.ndarray): Adjacency matrix [12, 12]
    """
    A = np.zeros((12, 12))
    edges = [
        (1, 2), (2, 0), (0, 3),             # head-thorax-tag-abdomen
        (1, 4), (4, 5),                     # left antenna
        (1, 6), (6, 7),                     # right antenna
        (1, 8), (1, 9),                     # mandibles
        (3, 10),                            # abdomen to tip
    ]
    for i, j in edges:
        A[i, j] = A[j, i] = 1
    return A

# === Model Definition ===
import torch
import torch.nn as nn
import torch.nn.functional as F

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, temporal_dropout=0.0):
        super().__init__()
        self.register_buffer("A", torch.tensor(A, dtype=torch.float32))
        self.temporal_dropout = temporal_dropout
        self.spatial = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.temporal = nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), padding=(4, 0))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        A = self.A.to(x.device)  # Ensure A is on the same device as x
        x = torch.matmul(A, x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        x = self.relu(self.bn(self.temporal(self.spatial(x))))
        if self.training and self.temporal_dropout > 0:
            x = F.dropout(x, p=self.temporal_dropout, training=True)
        return x


class STGCNModel(nn.Module):
    def __init__(self, A, in_channels=10, hidden=32, time_len=60, num_nodes=11, node_dropout_rate=0.2):
        super().__init__()
        self.node_dropout_rate = node_dropout_rate
        self.num_nodes = num_nodes

        self.gcn1 = STGCNBlock(in_channels, hidden, A)
        self.gcn2 = STGCNBlock(hidden, hidden, A)
        self.gcn3 = STGCNBlock(hidden, hidden, A)

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden * time_len * num_nodes, 1)

    def forward(self, x):
        if self.training and self.node_dropout_rate > 0:
            node_mask = (torch.rand(self.num_nodes, device=x.device) > self.node_dropout_rate).float()
            x = x * node_mask[None, None, None, :]

        x = self.gcn1(x)
        x = self.gcn2(x)
        x = self.gcn3(x)
        x = self.dropout(x)
        x = x.reshape(x.size(0), -1).contiguous()
        return self.fc(x)
        
def vote_for_nearest_target(locs, ego_idx, center_frame, window, track_names):
    """
    Uses the ego's mandible center and antenna tips to vote for the nearest body node
    (tag, head, thorax, abdomen) of other tracks over the middle 30 frames of the window.

    Returns:
        nearest_idx (int): Index of the voted target track.
        vote_counts (dict): Counter of how often each target track was closest.
        node_counts (dict): Counter of how often each body node (by name) was closest for the voted track.
        dominant_node (str): Most frequently nearest body node (e.g., "head").
    """
    import numpy as np
    from collections import Counter

    body_joints = [0, 1, 2, 3]  # tag, head, thorax, abdomen
    body_labels = {0: "tag", 1: "head", 2: "thorax", 3: "abdomen"}

    T, J, _, N = locs.shape
    start = center_frame - window // 2
    end = center_frame + window // 2
    if start < 0 or end > T or end - start < 30:
        return None, {}, {}, None

    # === Restrict to middle 30 frames
    mid_start = start + (window - 30) // 2
    mid_end = mid_start + 30
    frames = np.arange(mid_start, mid_end)

    try:
        # Ego reference points: mandible center, left tip, right tip
        l_mand = locs[frames, 8, :, ego_idx]
        r_mand = locs[frames, 9, :, ego_idx]
        mand_center = 0.5 * (l_mand + r_mand)

        l_tip = locs[frames, 5, :, ego_idx]
        r_tip = locs[frames, 7, :, ego_idx]
    except IndexError:
        return None, {}, {}, None

    if any(pt.ndim != 2 or pt.shape[0] != 30 for pt in [mand_center, l_tip, r_tip]):
        return None, {}, {}, None

    ref_points = [mand_center, l_tip, r_tip]
    ref_valid = [np.all(np.isfinite(p), axis=1) for p in ref_points]
    valid = ref_valid[0] | ref_valid[1] | ref_valid[2]

    # === Extract target body joints
    targets = locs[frames[:, None], body_joints, :, :]  # [30, 4, 2, N]
    isfinite = np.isfinite(targets).all(axis=2)
    mask = np.broadcast_to(isfinite[:, :, np.newaxis, :], targets.shape)
    targets = np.where(mask, targets, np.nan)

    vote_counts = Counter()
    node_counts = Counter()

    for ref, ref_mask in zip(ref_points, ref_valid):
        diffs = targets - ref[:, None, :, None]  # [30, 4, 2, N]
        dists = np.linalg.norm(diffs, axis=2)    # [30, 4, N]
        dists[:, :, ego_idx] = np.inf            # exclude ego track

        min_dists = np.nanmin(dists, axis=1)     # [30, N]
        closest_tracks = np.argmin(min_dists, axis=1)  # [30]

        for i, valid_frame in enumerate(ref_mask):
            if not valid_frame:
                continue
            tid = closest_tracks[i]
            if np.isinf(min_dists[i, tid]):
                continue
            vote_counts[tid] += 1

            closest_node = int(np.argmin(dists[i, :, tid]))
            node_counts[body_labels[closest_node]] += 1

    if not vote_counts:
        return None, {}, {}, None

    nearest_idx = vote_counts.most_common(1)[0][0]
    dominant_node = node_counts.most_common(1)[0][0] if node_counts else None

    return nearest_idx, vote_counts, node_counts, dominant_node

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import h5py
import numpy as np

def process_ego_track(args, augment=False):
    ego_idx, ego_name, locs, track_names, window, stride, frame_limit = args
    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    from collections import defaultdict

    def compute_angle(v1, v2):
        norm1 = np.linalg.norm(v1, axis=-1)
        norm2 = np.linalg.norm(v2, axis=-1)
        dot = np.einsum("...i,...i", v1, v2)
        cos_theta = dot / (norm1 * norm2 + 1e-6)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return np.arccos(cos_theta)

    T, J, _, N = locs.shape
    fused_windows, meta_info = [], []

    for start in range(0, frame_limit - window + 1, stride):
        center = start + window // 2
        if center >= T:
            continue

        ego_win = locs[start:start + window, :, :, ego_idx]
        ego_win = gaussian_filter1d(ego_win, sigma=0.1, axis=0)
        if not np.all(np.isfinite(ego_win)):
            continue

        target_idx, _, _, dominant_node = vote_for_nearest_target(locs, ego_idx, center, window, track_names)
        if target_idx is None or target_idx == ego_idx:
            continue

        target_tag = locs[start:start + window, 0, :, target_idx]
        target_tag = gaussian_filter1d(target_tag, sigma=0.1, axis=0)
        if not np.all(np.isfinite(target_tag)):
            continue

        vel = np.diff(ego_win, axis=0, prepend=ego_win[:1])
        acc = np.diff(vel, axis=0, prepend=vel[:1])  # acceleration

        tag = ego_win[0, 0, :]
        head = ego_win[0, 1, :]
        y_axis = head - tag
        if not np.all(np.isfinite(y_axis)) or np.linalg.norm(y_axis) == 0:
            continue
        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.array([-y_axis[1], y_axis[0]])
        R_align = np.stack([x_axis, y_axis], axis=1)

        ego_pos = (ego_win - tag) @ R_align
        vel = vel @ R_align
        acc = acc @ R_align
        target_tag_vel = np.diff(target_tag, axis=0, prepend=target_tag[:1])
        target_tag_pos = (target_tag - tag) @ R_align / 100.0
        target_tag_vel = target_tag_vel @ R_align / 100.0

        scale = 100.0
        ego_pos /= scale
        vel /= scale
        acc /= scale

        dists_to_target = np.linalg.norm(ego_pos - target_tag_pos[:, None, :], axis=-1, keepdims=True)

        # === Define edges
        edges = [
            (1, 2), (2, 0), (0, 3),
            (1, 4), (4, 5),
            (1, 6), (6, 7),
            (1, 8), (1, 9),
            (3, 10)
        ]
        node_neighbors = defaultdict(list)
        for i, j in edges:
            node_neighbors[i].append(j)
            node_neighbors[j].append(i)

        # === Node angle
        node_angles = np.zeros((window, J, 1))
        for n in range(J):
            neighbors = node_neighbors[n]
            if len(neighbors) >= 2:
                j, k = neighbors[:2]
                v1 = ego_pos[:, j] - ego_pos[:, n]
                v2 = ego_pos[:, k] - ego_pos[:, n]
                node_angles[:, n, 0] = compute_angle(v1, v2)

        # === Angular velocity
        angle_vel = np.diff(node_angles[:, :, 0], axis=0, prepend=node_angles[:1, :, 0])[..., None]

        # === Edge length
        node_edge_lengths = np.zeros((window, J, 1))
        edge_counts = np.zeros(J)
        for i, j in edges:
            length = np.linalg.norm(ego_pos[:, i] - ego_pos[:, j], axis=-1, keepdims=True)
            node_edge_lengths[:, i] += length
            node_edge_lengths[:, j] += length
            edge_counts[i] += 1
            edge_counts[j] += 1
        for n in range(J):
            if edge_counts[n] > 0:
                node_edge_lengths[:, n] /= edge_counts[n]

        # === Final feature stack: 10 channels
        ego_feat = np.concatenate([
            ego_pos,             # 2
            vel,                 # 2
            acc,                 # 2
            dists_to_target,     # 1
            node_angles,         # 1
            angle_vel,           # 1
            node_edge_lengths    # 1
        ], axis=-1)

        target_dist = np.linalg.norm(tag - target_tag, axis=-1, keepdims=True)
        target_tag_feat = np.concatenate([
            target_tag_pos,                     # 2
            target_tag_vel,                    # 2
            np.zeros((window, 2)),             # Acceleration placeholder
            target_dist,                       # 1
            np.zeros((window, 3))              # Angle, angle_vel, edge length
        ], axis=-1)  # Total: 10 channels
        
        fused_feat = np.concatenate([ego_feat, target_tag_feat[:, None, :]], axis=1)
        if fused_feat.shape[-1] != 10 or fused_feat.shape[1] != 12:
            print(f"❌ Skipping malformed window: shape={fused_feat.shape}, ego={ego_name}, frame={start}")
            continue
        
        fused_windows.append(fused_feat)
        meta_info.append((ego_name, track_names[target_idx], start, dominant_node))

        # === Augmentation
        if augment:
            flip = lambda x: x.copy()
            flip[..., 0] *= -1
            fused_feat_flip = fused_feat.copy()
            fused_feat_flip[:, :, 0] *= -1  # flip pos-x
            fused_feat_flip[:, :, 2] *= -1  # flip vel-x
            fused_feat_flip[:, :, 4] *= -1  # flip acc-x
            fused_feat_flip[:, :, 7] *= -1  # flip target-pos-x
            fused_feat_flip[:, :, 8] *= -1  # flip target-vel-x
            fused_windows.append(fused_feat_flip)
            meta_info.append((ego_name, track_names[target_idx], start, dominant_node))

    return fused_windows, meta_info

def parallel_extract_ego_windows(h5_file_path, window=60, stride=15, frame_limit=108000, cache_dir=None, augment=False):
    import h5py, numpy as np, multiprocessing
    from functools import partial
    from concurrent.futures import ProcessPoolExecutor

    with h5py.File(h5_file_path, "r") as f:
        raw = f["tracks"][:].T
        true_frame_count = raw.shape[0]
        used_limit = min(frame_limit, true_frame_count)
        if true_frame_count < frame_limit:
            print(f"⚠️ WARNING: file has only {true_frame_count} frames (less than frame_limit={frame_limit})")
        locs = raw[:used_limit]  # [T, J, 2, N]

        track_names = [n.decode().strip() for n in f["track_names"][:]]

    args_list = [(i, track_names[i], locs, track_names, window, stride, frame_limit)
                 for i in range(len(track_names))]

    expected_shape = (window, 12, 10)
    ego_all, meta_all = [], []

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        process_func = partial(process_ego_track, augment=augment)
        for ego_windows, meta in executor.map(process_func, args_list):
            for win, meta_info in zip(ego_windows, meta):
                if win.shape != expected_shape:
                    print(f"⚠️ Skipping window with shape {win.shape} (expected {expected_shape})")
                    continue
                ego_all.append(win)
                meta_all.append(meta_info)

    if len(ego_all) == 0:
        print("⚠️ No valid ego windows found.")
        return np.empty((0, window, 12, 10)), []

    X = np.stack(ego_all)
    return X, meta_all
    
def generate_ego_windows(h5_file_path, window=60, stride=15, frame_limit=18000, augment=True):
    X_ego, meta_info = parallel_extract_ego_windows(
        h5_file_path,
        window=window,
        stride=stride,
        frame_limit=frame_limit,
        augment=augment
    )
    return X_ego, meta_info
    
def prepare_dataset(pairs, window=60, stride=15, frame_limit=18000, cache_dir=None):
    import multiprocessing
    import os
    import numpy as np
    from collections import Counter

    print(f"🔄 Preparing dataset using multiprocessing ({multiprocessing.cpu_count()} cores)...")
    X_all, y_all = [], []
    ego_names = []
    flip_flags = []
    original_positive_counts = Counter()
    flipped_positive_counts = Counter()

    for h5_path, label_path in pairs:
        print(f"📂 Processing: {os.path.basename(h5_path)}")

        # === Extract ego windows
        X_ego, meta_info = parallel_extract_ego_windows(
            h5_file_path=h5_path,
            window=window,
            stride=stride,
            frame_limit=frame_limit,
            cache_dir=cache_dir,
            augment=False  # ensure no flip happens here
        )
        if len(meta_info) == 0:
            continue

        # === Parse duel labels
        labels = parse_duel_labels(label_path)
        duel_map = {ego: set(ts) for ego, ts in labels.items()}

        y_file = []
        ego_list = []
        X_augmented = []
        flip_flags_file = []

        for i, (ego_name, target_name, start_frame, dominant_node) in enumerate(meta_info):
            center = start_frame + window // 2
            window_data = X_ego[i]
            is_positive = ego_name in duel_map and center in duel_map[ego_name]

            # === Always include original window
            X_augmented.append(window_data)
            y_file.append(1 if is_positive else 0)
            ego_list.append(ego_name)
            flip_flags_file.append(0)  # original window

            if is_positive:
                original_positive_counts[ego_name] += 1

                # === Flip duplicate
                flipped = window_data.copy()
                flipped[:, :, 0] *= -1
                X_augmented.append(flipped)
                y_file.append(1)
                ego_list.append(ego_name)
                flip_flags_file.append(1)  # flipped window
                flipped_positive_counts[ego_name] += 1

        X_all.append(np.stack(X_augmented))
        y_all.append(np.array(y_file, dtype=np.uint8))
        ego_names.extend(ego_list)
        flip_flags.extend(flip_flags_file)

    if not X_all:
        print("⚠️ No training windows found.")
        return np.empty((0, window, 12, 10)), np.array([]), []

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    flip_flags = np.array(flip_flags, dtype=np.uint8)

    total_positives = int(np.sum(y_all))
    print(f"\n📊 Total positive windows (including flipped): {total_positives}")

    print("📊 Per-ego track positive counts:")
    for name in sorted(set(original_positive_counts) | set(flipped_positive_counts)):
        orig = original_positive_counts[name]
        flip = flipped_positive_counts[name]
        print(f"  - {name}: {orig} original + {flip} flipped = {orig + flip}")

    return X_all, y_all, ego_names

def evaluate_event_matching(df, loaded_data, label="TEST", verbose=True):
    df = df.copy()

    if "Time_sec" not in df.columns:
        df["Time_sec"] = df["Time"].apply(time_to_seconds)

    df = df[df["Frame"] < FRAME_LIMIT].copy()  # 🧱 Enforce frame limit
    df["Label"] = "FP"
    total_tp, total_fp, total_fn, total_gt = 0, 0, 0, 0
    match_radius = 2  # seconds
    matched_gt_global = set()

    for _, _, val_file in loaded_data:
        if val_file is None or not os.path.exists(val_file):
            continue

        gt_df = pd.read_csv(val_file, header=None)
        gt_map = {}
        for _, row in gt_df.iterrows():
            track = str(row.iloc[0]).strip()
            times = [time_to_seconds(t) for t in row.iloc[1:] if pd.notna(t)]
            filtered = [t for t in times if t is not None and t * 30 < FRAME_LIMIT]
            gt_map[track] = sorted(filtered)
            total_gt += len(filtered)

        preds = df[(df["ValidationFile"] == val_file) & (df["Duel"] == 1)].copy()
        matched_gt = set()
        labels = []

        for i, row in preds.iterrows():
            track = str(row.iloc[0]).strip()  # GT side
            pred_ts = row["Time_sec"]
            is_tp = False
            for gt_ts in gt_map.get(track, []):
                if abs(pred_ts - gt_ts) <= match_radius and (track, gt_ts) not in matched_gt:
                    matched_gt.add((track, gt_ts))
                    matched_gt_global.add((track, gt_ts))
                    is_tp = True
                    break
            labels.append("TP" if is_tp else "FP")

        df.loc[preds.index, "Label"] = labels
        total_tp += labels.count("TP")
        total_fp += labels.count("FP")

        for track, gt_times in gt_map.items():
            for gt in gt_times:
                if (track, gt) not in matched_gt:
                    fn_row = {
                        "Track": track,
                        "Time": f"{gt // 60}:{gt % 60:02d}",
                        "Frame": gt * 30,
                        "Probability": 0.0,
                        "Duel": 0,
                        "ValidationFile": val_file,
                        "Time_sec": gt,
                        "Label": "FN"
                    }
                    df = pd.concat([df, pd.DataFrame([fn_row])], ignore_index=True)
                    total_fn += 1

    if total_tp + total_fn != total_gt:
        print(f"⚠️ Warning: TP+FN={total_tp + total_fn} ≠ total_GT={total_gt} (Δ={total_gt - (total_tp + total_fn)})")

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    if verbose:
        print(f"\n📊 Evaluation on {label} SET (event-matching ±{match_radius}s):")
        print(f"   TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1 Score:  {f1:.4f}")

    return precision, recall, f1, df

def cluster_by_peaks(df, distance=3, height=0.1):
    """
    Clusters overlapping windows and finds all peaks within each overlapping chunk.

    Args:
        df (pd.DataFrame): Prediction DataFrame with 'Track', 'Frame', 'Probability', etc.
        distance (int): Minimum distance between peaks in a chunk (in frames).
        height (float): Minimum probability height to qualify as a peak.

    Returns:
        pd.DataFrame: Subset of `df` with only local peaks retained within overlapping chunks.
    """
    from scipy.signal import find_peaks

    if "CenterFrame" not in df.columns:
        df["CenterFrame"] = df["Frame"] + WINDOWSIZE // 2

    clustered = []

    for track, group in df.groupby("Track"):
        group = group.sort_values("CenterFrame").reset_index(drop=True)
        frames = group["CenterFrame"].values
        probs = group["Probability"].values

        overlaps = [0]
        for i in range(1, len(group)):
            if frames[i] <= frames[i - 1] + WINDOWSIZE:  # overlapping chunk
                overlaps.append(i)
            else:
                chunk = group.iloc[overlaps]
                chunk_probs = chunk["Probability"].values
                peaks, _ = find_peaks(chunk_probs, distance=distance, height=height)
                if len(peaks) > 0:
                    clustered.append(chunk.iloc[peaks])
                overlaps = [i]

        # Final chunk
        if overlaps:
            chunk = group.iloc[overlaps]
            chunk_probs = chunk["Probability"].values
            peaks, _ = find_peaks(chunk_probs, distance=distance, height=height)
            if len(peaks) > 0:
                clustered.append(chunk.iloc[peaks])

    return pd.concat(clustered, ignore_index=True) if clustered else df.head(0).copy()

from torch.utils.data import DataLoader, TensorDataset

def predict_in_batches(model, X_tensor, batch_size=256, device=None):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    model.eval()
    preds = []
    loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size)
    with torch.no_grad():
        for xb in loader:
            xb = xb[0].to(device)
            preds.append(torch.sigmoid(model(xb)).squeeze().cpu())
    return torch.cat(preds).numpy()

def train_stgcn_model(pairs, model_dir, device=None):
    import os, json, random, torch, numpy as np, matplotlib.pyplot as plt
    from sklearn.utils import resample
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
    from torch.utils.data import DataLoader, TensorDataset
    from collections import defaultdict
    import pandas as pd
    import torch.nn.functional as F

    os.makedirs(model_dir, exist_ok=True)
    config_path = os.path.join(model_dir, "model_config.json")

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        window = config.get("window_size", 60)
        stride = config.get("stride", 15)
        distance = config.get("cluster_distance", 3)
        height = config.get("height", 0.10)
        print(f"Loaded config: window={window}, stride={stride}, cluster_distance={distance}")
    else:
        window, stride, distance = 60, 15, 3
        height = 0.1
        print("No config found. Using default parameters.")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    positive_counts = defaultdict(int)
    negative_counts = defaultdict(int)
    global_pos_pool = []
    global_pos_meta = []
    X_neg_all, y_neg_all, meta_neg_all = [], [], []

    for h5_path, label_path in pairs:
        X_file, y_file, ego_names = prepare_dataset([(h5_path, label_path)], window=window, stride=stride)
        if len(y_file) == 0:
            continue

        pos_idx = [i for i, val in enumerate(y_file) if val == 1]
        neg_idx = [i for i, val in enumerate(y_file) if val == 0]

        for i in pos_idx:
            ego = ego_names[i]
            positive_counts[ego] += 1
        for i in neg_idx:
            ego = ego_names[i]
            negative_counts[ego] += 1

        global_pos_pool.extend([X_file[i] for i in pos_idx])
        global_pos_meta.extend([(ego_names[i], i * stride, label_path) for i in pos_idx])

        n_neg_sample = max(1, int(len(neg_idx) * 0.2))
        sampled_neg_idx = random.sample(neg_idx, n_neg_sample)

        resampled_neg = defaultdict(int)
        for i in sampled_neg_idx:
            ego = ego_names[i]
            resampled_neg[ego] += 1

        print(f"\nFile: {os.path.basename(label_path)} — Sampled Negatives:")
        for ego in sorted(set(ego_names)):
            print(f"  - {ego}: {resampled_neg.get(ego, 0)} negative")

        X_neg_all.extend([X_file[i] for i in sampled_neg_idx])
        y_neg_all.extend([0] * len(sampled_neg_idx))
        meta_neg_all.extend([(ego_names[i], i * stride, label_path) for i in sampled_neg_idx])

    total_neg = len(X_neg_all)
    total_pos = len(global_pos_pool)

    if total_pos == 0:
        raise ValueError("No positive samples found across dataset!")

    print(f"\n📊 Before Sampling → Available Positives: {total_pos}, Negatives to Match: {total_neg}")

    sampled_pos_idx = resample(range(total_pos), replace=True, n_samples=total_neg, random_state=42)

    X_pos_final = [global_pos_pool[i] for i in sampled_pos_idx]
    y_pos_final = [1] * total_neg
    meta_pos_final = [global_pos_meta[i] for i in sampled_pos_idx]

    print(f"\nGlobal sampling → Total Positives: {len(X_pos_final)}, Total Negatives: {total_neg}")

    X_all = X_pos_final + X_neg_all
    y_all = y_pos_final + y_neg_all
    meta_info = meta_pos_final + meta_neg_all

    combined = list(zip(X_all, y_all, meta_info))
    random.shuffle(combined)
    X_all, y_all, meta_info = zip(*combined)

    X_tensor = torch.tensor(np.stack(X_all), dtype=torch.float32).permute(0, 3, 1, 2)  # [B, 4, T, N]
    y_tensor = torch.tensor(y_all, dtype=torch.float32)
    
    train_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=64, shuffle=True)

    A = build_adjacency_matrix_with_target()
    np.save(os.path.join(model_dir, "adj_matrix.npy"), A)
    model = STGCNModel(
        A=A,
        in_channels=10,
        hidden=32,
        time_len=window,
        num_nodes=12,
        node_dropout_rate=0.2
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def smooth_bce_loss(logits, targets, smoothing=0.1):
        with torch.no_grad():
            targets = targets * (1 - smoothing) + 0.5 * smoothing
        return F.binary_cross_entropy_with_logits(logits, targets)

    best_loss = float("inf")
    patience = 0
    max_patience = 10
    auc_history = []

    for epoch in range(500):
        model.train()
        total_loss = 0
    
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
        
            logits = model(xb).squeeze()
            loss = smooth_bce_loss(logits, yb)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            total_loss += loss.item() * xb.size(0)
    
        # ✅ Log Grad × Input contribution per channel (now includes 10 channels
        log_channel_contributions_epoch(model, train_loader, device)
        
        avg_loss = total_loss / len(train_loader.dataset)
        
        # === Evaluation on full training set
        model.eval()
        with torch.no_grad():
            y_probs = predict_in_batches(model, X_tensor, batch_size=256, device=device)
            y_true = y_tensor.detach().cpu().numpy()
            auc = roc_auc_score(y_true, y_probs)
            y_pred = (y_probs >= 0.5).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            auc_history.append(auc)
        
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}")
        
        # === Select top confident windows to analyze node importance
        with torch.no_grad():
            logits_full = []
            loader = DataLoader(TensorDataset(X_tensor), batch_size=256)
            for xb in loader:
                xb = xb[0].to(device)
                logits_full.append(model(xb).squeeze().cpu())
            logits_full = torch.cat(logits_full)

        probs_full = torch.sigmoid(logits_full)
        top_indices = torch.topk(probs_full, k=min(64, X_tensor.size(0)), sorted=False).indices
        X_eval = X_tensor[top_indices].detach().clone().to(device).requires_grad_(True)
        X_eval.retain_grad()

        output = model(X_eval).squeeze()
        output.sum().backward()

        grad = X_eval.grad.detach()  # [B, 10, T, 12]
        contrib = (grad * X_eval).abs().mean(dim=(0, 2))  # [10, 12]

        # === Split contributions by channel
        pos_contrib      = contrib[0:2].sum(dim=0).detach().cpu().numpy()  # [12]
        vel_contrib      = contrib[2:4].sum(dim=0).detach().cpu().numpy()  # [12]
        acc_contrib      = contrib[4:6].sum(dim=0).detach().cpu().numpy()  # [12]
        dist_contrib     = contrib[6].detach().cpu().numpy()               # [12]
        angle_contrib    = contrib[7].detach().cpu().numpy()               # [12]
        angle_vel_contrib= contrib[8].detach().cpu().numpy()               # [12]
        length_contrib   = contrib[9].detach().cpu().numpy()               # [12]

        node_names = [
            "ego_tag", "ego_head", "ego_thorax", "ego_abdomen",
            "ego_l_antenna_joint", "ego_l_antenna_tip",
            "ego_r_antenna_joint", "ego_r_antenna_tip",
            "ego_l_mandible", "ego_r_mandible", "ego_abdomen_tip",
            "target_tag"
        ]

        print(f"\n📊 [Epoch {epoch+1} Node-wise Contributions]")
        for label, contrib_array in zip(
            ["Position", "Velocity", "Acceleration", "Distance", "NodeAngle", "AngleVel", "EdgeLength"],
            [pos_contrib, vel_contrib, acc_contrib, dist_contrib, angle_contrib, angle_vel_contrib, length_contrib]
        ):
            print(f"📌 {label}:")
            for i in range(11):  # Exclude target_tag from ego node summary
                print(f"  {node_names[i]:18s} → {label[:4]}: {contrib_array[i]:.6f}")

        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            patience = 0
            torch.save(model.state_dict(), os.path.join(model_dir, "stgcn_duel_model.pt"))
            print(f"✅ New best loss: {best_loss:.4f} → model saved.")
        else:
            patience += 1
            print(f"⏳ No improvement for {patience} epoch(s).")
            if patience >= max_patience:
                print("🛑 Early stopping.")
                break
    
    print(f"🏁 Final best loss: {best_loss:.4f}")


    # === Compute and print window-level metrics at threshold 0.5
    y_true = y_tensor.detach().cpu().numpy()
    y_pred = (y_probs >= 0.5).astype(int)
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"\n📏 Window-Level Metrics @ Threshold 0.5:")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1 Score:  {f1:.4f}")

    # === Plot AUC history
    plt.figure()
    plt.plot(auc_history, marker="o", label="Train AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC Score")
    plt.title("Training AUC Over Epochs")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "train_auc_curve.png"))
    plt.close()

    # === Save config with metrics
    config = {
        "window_size": window,
        "stride": stride,
        "cluster_distance": distance,
        "train_auc": round(auc_history[-1], 4),
        "window_precision": round(precision, 4),
        "window_recall": round(recall, 4),
        "window_f1": round(f1, 4),
        "height": round(height, 4),
        "in_channels": 10,
        "hidden": 32,
        "node_dropout_rate": 0.2
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Final config saved to: {config_path}")
    
def extract_and_predict(h5_file, label_file, model, device, window, stride, frame_limit, augment=False):
    import torch
    import h5py
    import numpy as np
    import pandas as pd
    from torch.utils.data import DataLoader, TensorDataset

    # === Extract ego + target fused windows
    X_ego, meta_info = generate_ego_windows(
        h5_file,
        window=window,
        stride=stride,
        frame_limit=frame_limit,
        augment=augment
    )
    if X_ego is None or len(X_ego) == 0:
        return None, None

    # === Convert to tensor and permute to [B, C, T, N]
    X_tensor = torch.from_numpy(X_ego).float().permute(0, 3, 1, 2)  # [B, 4, T, J]

    expected_channels = 10
    expected_nodes = model.A.shape[0] if hasattr(model, "A") else X_tensor.shape[-1]
    if X_tensor.shape[1:] != (expected_channels, window, expected_nodes):
        raise ValueError(f"⚠️ Input tensor shape mismatch: got {X_tensor.shape}, expected (*, 10, {window}, {expected_nodes})")

    # === Batch prediction
    with torch.no_grad():
        loader = DataLoader(TensorDataset(X_tensor), batch_size=64)
        probas = torch.cat([
            torch.sigmoid(model(xb[0].to(device))).squeeze().cpu()
            for xb in loader
        ]).numpy()

    # === Load track names to index map
    with h5py.File(h5_file, "r") as f:
        track_names = [n.decode().strip() for n in f["track_names"][:]]
    track_to_index = {name: i for i, name in enumerate(track_names)}

    # === Create DataFrame with metadata (including dominant_node)
    df = pd.DataFrame(meta_info, columns=["Track", "Target", "Frame", "DominantNode"])
    df["TrackIndex"] = df["Track"].map(track_to_index)
    df["Probability"] = probas
    df["ValidationFile"] = label_file
    df["CenterFrame"] = df["Frame"] + window // 2
    df["Time_sec"] = df["CenterFrame"] / 30
    df["Time"] = (df["Time_sec"] // 60).astype(int).astype(str) + ":" + (df["Time_sec"] % 60).astype(int).astype(str).str.zfill(2)

    return df[df["Frame"] < frame_limit].copy(), probas
    
def log_channel_contributions_epoch(model, dataloader, device):
    import torch

    model.eval()
    total_contrib = None
    count = 0

    for xb, yb in dataloader:
        xb = xb.clone().detach().to(device).requires_grad_(True)
        xb.retain_grad()

        output = model(xb).squeeze()
        output.mean().backward()

        grad = xb.grad.detach()  # [B, C, T, N]
        contrib = (grad * xb).abs().mean(dim=(0, 2, 3))  # [C]

        if total_contrib is None:
            total_contrib = torch.zeros_like(contrib)

        total_contrib += contrib
        count += 1

    avg_contrib = total_contrib / count

    channel_names = [
        "Pos-X", "Pos-Y",
        "Vel-X", "Vel-Y",
        "Acc-X", "Acc-Y",
        "Dist",
        "NodeAngle",
        "AngleVel",
        "EdgeLength"
    ]

    print(f"\n📊 [Epoch Channel Contribution]")
    for i, name in enumerate(channel_names[:len(avg_contrib)]):
        print(f"  {name}: {avg_contrib[i].item():.6f}")
    
def compute_node_importance(model, X_input, model_dir, top_k=64):
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    # === Node names: 11 ego (exclude target_tag for plotting)
    ego_parts = [
        "tag", "head", "thorax", "abdomen", "l_antenna_joint", "l_antenna_tip",
        "r_antenna_joint", "r_antenna_tip", "l_mandible", "r_mandible", "abdomen_tip"
    ]
    NODE_NAMES = [f"ego_{name}" for name in ego_parts]

    model.eval()

    # === Step 1: Get confident predictions
    with torch.no_grad():
        logits = model(X_input).squeeze()
        probs = torch.sigmoid(logits)
        top_indices = torch.topk(probs, k=min(top_k, X_input.size(0)), sorted=False).indices

    # === Step 2: Enable gradient for input and rerun selected batch
    selected_input = X_input[top_indices].clone().detach().requires_grad_(True)
    output = model(selected_input).squeeze()
    output.sum().backward()

    grad = selected_input.grad.detach()  # [B, 10, T, 12]
    contrib = (grad * selected_input).abs().mean(dim=(0, 2))  # [10, 12]

    # === Extract per-channel contributions (truncate to ego nodes)
    pos_contrib     = contrib[0:2].sum(dim=0).detach().cpu().numpy()[:11]  # Pos-X, Pos-Y
    vel_contrib     = contrib[2:4].sum(dim=0).detach().cpu().numpy()[:11]  # Vel-X, Vel-Y
    acc_contrib     = contrib[4:6].sum(dim=0).detach().cpu().numpy()[:11]  # Acc-X, Acc-Y
    dist_contrib    = contrib[6].detach().cpu().numpy()[:11]               # Distance to target
    angle_contrib   = contrib[7].detach().cpu().numpy()[:11]               # Node angle
    anglevel_contrib= contrib[8].detach().cpu().numpy()[:11]               # Angular velocity
    length_contrib  = contrib[9].detach().cpu().numpy()[:11]               # Edge length

    # === Print summary
    print("\n🔍 Node-wise Grad × Input contributions:")
    for i, name in enumerate(NODE_NAMES):
        print(
            f"  {name:18s} → Pos: {pos_contrib[i]:.6f}  Vel: {vel_contrib[i]:.6f}  Acc: {acc_contrib[i]:.6f}  "
            f"Dist: {dist_contrib[i]:.6f}  Angle: {angle_contrib[i]:.6f}  dAngle: {anglevel_contrib[i]:.6f}  "
            f"Length: {length_contrib[i]:.6f}"
        )

    # === Plot grouped bar chart
    x = np.arange(len(NODE_NAMES))
    width = 0.12

    plt.figure(figsize=(18, 5))
    plt.bar(x - 3*width, pos_contrib, width, label='Position')
    plt.bar(x - 2*width, vel_contrib, width, label='Velocity')
    plt.bar(x - width,   acc_contrib, width, label='Acceleration')
    plt.bar(x,           dist_contrib, width, label='Distance')
    plt.bar(x + width,   angle_contrib, width, label='Node Angle')
    plt.bar(x + 2*width, anglevel_contrib, width, label='Angle Velocity')
    plt.bar(x + 3*width, length_contrib, width, label='Edge Length')

    plt.xticks(x, NODE_NAMES, rotation=45, ha='right')
    plt.xlabel("Node")
    plt.ylabel("Importance (Grad × Input)")
    plt.title("Per-Node Importance by Feature Type")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(model_dir, "node_importance_grouped.png")
    plt.savefig(out_path)
    plt.close()
    print(f"📊 Grouped node importance plot saved to: {out_path}")

    return pos_contrib, vel_contrib, acc_contrib, dist_contrib, angle_contrib, anglevel_contrib, length_contrib

def evaluate_and_plot_stgcn(train_pairs, test_pairs, model_dir):
    import os, json, torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
    from tqdm import tqdm

    FRAME_LIMIT = 18000
    plot_dir = os.path.join(model_dir, "model_evaluation_plot")
    pred_dir = os.path.join(model_dir, "labeled_predictions")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    # === Load config and model
    config_path = os.path.join(model_dir, "model_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    window = config.get("window_size", 60)
    stride = config.get("stride", 15)
    hidden = config.get("hidden", 32)
    in_channels = config.get("in_channels", 10)
    node_dropout_rate = config.get("node_dropout_rate", 0.2)

    A = np.load(os.path.join(model_dir, "adj_matrix.npy"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = STGCNModel(
        A=A, in_channels=in_channels, hidden=hidden,
        time_len=window, num_nodes=A.shape[0],
        node_dropout_rate=node_dropout_rate
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, "stgcn_duel_model.pt"), map_location=device))
    model.eval()

    def get_clustered_predictions(pairs):
        dfs = []
        for h5_file, label_file in tqdm(pairs, desc="🔍 Predicting and Clustering"):
            df, _ = extract_and_predict(h5_file, label_file, model, device, window, stride, FRAME_LIMIT, augment=False)
            if df is not None and not df.empty:
                df = cluster_by_peaks(df, distance=3, height=0.1)
                df["ValidationFile"] = label_file
                dfs.append(df)
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    clustered_train_df = get_clustered_predictions(train_pairs)
    clustered_test_df = get_clustered_predictions(test_pairs)

    thresholds = np.linspace(0.01, 0.99, 50)
    train_f1s, test_f1s = [], []
    eval_train = [(None, None, f) for _, f in train_pairs]
    eval_test = [(None, None, f) for _, f in test_pairs]

    for t in tqdm(thresholds, desc="📈 Evaluating thresholds"):
        train_cut = clustered_train_df[clustered_train_df["Probability"] >= t].copy()
        test_cut = clustered_test_df[clustered_test_df["Probability"] >= t].copy()
        train_cut["Duel"] = 1
        test_cut["Duel"] = 1
        train_f1s.append(evaluate_event_matching(train_cut, eval_train, verbose=False)[2])
        test_f1s.append(evaluate_event_matching(test_cut, eval_test, verbose=False)[2])

    avg_f1s = [(tr + te) / 2 for tr, te in zip(train_f1s, test_f1s)]
    best_idx = int(np.argmax(avg_f1s))
    best_thresh = float(thresholds[best_idx])

    def save_labeled_predictions(clustered_df, pairs, best_thresh):
        all_filtered = []
        for _, label_file in pairs:
            file_df = clustered_df[clustered_df["ValidationFile"] == label_file].copy()
            file_df = file_df[file_df["Probability"] >= best_thresh].copy()
            if file_df.empty:
                continue
    
            file_df["Duel"] = 1
            file_df["Label"] = "FP"
    
            # ✅ capture the updated dataframe
            _, _, _, updated_df = evaluate_event_matching(file_df, [(None, None, label_file)], verbose=False)
    
            df_out = updated_df[["Track", "Target", "Frame", "Time", "Probability", "Label", "DominantNode"]].copy()
            outname = os.path.splitext(os.path.basename(label_file))[0] + "_labeled.csv"
            df_out.to_csv(os.path.join(pred_dir, outname), index=False)
    
            all_filtered.append(updated_df)
    
        return pd.concat(all_filtered, ignore_index=True) if all_filtered else pd.DataFrame()

    train_df = save_labeled_predictions(clustered_train_df, train_pairs, best_thresh)
    test_df = save_labeled_predictions(clustered_test_df, test_pairs, best_thresh)

    print(f"\n🌟 Best Average F1: {(train_f1s[best_idx] + test_f1s[best_idx]) / 2:.4f} @ Threshold: {best_thresh:.3f}")
    print(f"   ⏺ Train F1 (from sweep): {train_f1s[best_idx]:.4f}")
    print(f"   ⏺ Test  F1 (from sweep): {test_f1s[best_idx]:.4f}")

    print("\n📊 Evaluation on TRAINING SET (event-matching ±2s):")
    evaluate_event_matching(train_df, eval_train, label="TRAIN", verbose=True)

    print("\n📁 Per-File Evaluation (Training Set):")
    for _, val_file in train_pairs:
        file_cut = train_df[train_df["ValidationFile"] == val_file].copy()
        file_cut["Duel"] = 1
        print(f"\n📄 File: {os.path.basename(val_file)}")
        evaluate_event_matching(file_cut, [(None, None, val_file)], label=os.path.basename(val_file), verbose=True)

    print("\n📊 Evaluation on TEST SET (event-matching ±2s):")
    evaluate_event_matching(test_df, eval_test, label="TEST", verbose=True)

    print("\n📁 Per-File Evaluation (Test Set):")
    for _, val_file in test_pairs:
        file_cut = test_df[(test_df["ValidationFile"] == val_file) & (test_df["Duel"] == 1)].copy()
        print(f"\n📄 File: {os.path.basename(val_file)}")
        evaluate_event_matching(file_cut, [(None, None, val_file)], label=os.path.basename(val_file), verbose=True)

    plt.figure()
    plt.plot(thresholds, train_f1s, label="Train F1", marker="o")
    plt.plot(thresholds, test_f1s, label="Test F1", marker="x")
    plt.axvline(best_thresh, color="red", linestyle="--", label=f"Best @ {best_thresh:.2f}")
    plt.xlabel("Threshold")
    plt.ylabel("Event-level F1 Score")
    plt.title("Event-level F1 vs Threshold")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "f1_vs_threshold.png"))
    plt.close()

    test_h5 = test_pairs[0][0]
    X_eval, _ = generate_ego_windows(test_h5, window, stride, FRAME_LIMIT, augment=False)
    if X_eval is not None and len(X_eval) > 0:
        X_tensor_all = torch.tensor(X_eval, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(X_tensor_all).squeeze())
            top_indices = torch.topk(probs, k=min(64, X_tensor_all.shape[0]), sorted=False).indices
        compute_node_importance(model, X_tensor_all[top_indices], plot_dir)
    else:
        print("⚠️ No windows found for node importance.")

    config["best_cutoff"] = round(best_thresh, 4)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n💾 Best cutoff saved to config: {config_path}")

# === Main Function ===
def main_train_and_evaluate(train_pairs, test_pairs, model_dir):
    import torch, os, json
    os.makedirs(model_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("🚀 Starting ST-GCN Training")
    train_stgcn_model(train_pairs, model_dir, device=device)

    print("🔁 Loading trained model")
    config_path = os.path.join(model_dir, "model_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    A = np.load(os.path.join(model_dir, "adj_matrix.npy"))
    model = STGCNModel(
        A=A,
        in_channels=config.get("in_channels", 10),
        hidden=config.get("hidden", 32),
        time_len=config.get("window_size", 60),
        num_nodes=A.shape[0],
        node_dropout_rate=config.get("node_dropout_rate", 0.2)
    ).to(device)

    model.load_state_dict(torch.load(os.path.join(model_dir, "stgcn_duel_model.pt"), map_location=device))
    model.eval()

    print("🔍 Evaluating and plotting on test set")
    evaluate_and_plot_stgcn(train_pairs, test_pairs, model_dir)

if __name__ == "__main__":
    # === Define Training and Testing File Pairs ===
    train_non_feeding = [
        ("/scratch/sl9623/container/prediction/h5_files/filtered_TS67/20250120/basler_record_20250120_07_filtered.h5",
         "/scratch/sl9623/container/duel_tracker/Duel_GT_TS67/Duel_label_AF24_plus/20250120_07_10min_dueling_label.csv"),
        ("/scratch/sl9623/container/prediction/h5_files/filtered_TS67/20250121/basler_record_20250121_12_filtered.h5",
         "/scratch/sl9623/container/duel_tracker/Duel_GT_TS67/Duel_label_AF24_plus/20250121_12_10min_dueling_label.csv"),
        ("/scratch/sl9623/container/prediction/h5_files/filtered_TS67/20250123/basler_record_20250123_12_filtered.h5",
         "/scratch/sl9623/container/duel_tracker/Duel_GT_TS67/Duel_label_AF24_plus/20250123_12_10min_dueling_label.csv"),
        ("/scratch/sl9623/container/prediction/h5_files/filtered_TS67/20250126/basler_record_20250126_12_filtered.h5",
         "/scratch/sl9623/container/duel_tracker/Duel_GT_TS67/Duel_label_AF24_plus/20250126_12_10min_dueling_label.csv"),
        ("/scratch/sl9623/container/prediction/h5_files/filtered_TS67/20250127/basler_record_20250127_12_filtered.h5",
         "/scratch/sl9623/container/duel_tracker/Duel_GT_TS67/Duel_label_AF24_plus/20250127_12_10min_dueling_label.csv"),
        ("/scratch/sl9623/container/prediction/h5_files/filtered_TS67/20250204/basler_record_20250204_12_filtered.h5",
         "/scratch/sl9623/container/duel_tracker/Duel_GT_TS67/Duel_label_AF24_plus/20250204_12_10min_dueling_label.csv"),
        ("/scratch/sl9623/container/prediction/h5_files/filtered_TS67/20250207/basler_record_20250207_12_filtered.h5",
         "/scratch/sl9623/container/duel_tracker/Duel_GT_TS67/Duel_label_AF24_plus/20250207_12_10min_dueling_label.csv"),
        ("/scratch/sl9623/container/prediction/h5_files/filtered_TS67/20250213/basler_record_20250213_11_filtered.h5",
         "/scratch/sl9623/container/duel_tracker/Duel_GT_TS67/Duel_label_AF24_plus/20250213_11_10min_dueling_label.csv")
    ]
    
    test_non_feeding = [
        ("/scratch/sl9623/container/prediction/h5_files/filtered_TS67/20250121/basler_record_20250121_16_filtered.h5",
         "/scratch/sl9623/container/duel_tracker/Duel_GT_TS67/Duel_label_AF24_plus/20250121_16_10min_dueling_label.csv")
    ]
    
    train_24AF = [
        ("/scratch/sl9623/container/prediction/h5_files/filtered_TS67/20250122/basler_record_20250122_16_filtered.h5",
         "/scratch/sl9623/container/duel_tracker/Duel_GT_TS67/Duel_label_AF24/20250122_16_10min_dueling_label.csv"),
        ("/scratch/sl9623/container/prediction/h5_files/filtered_TS67/20250123/basler_record_20250123_10_filtered.h5",
         "/scratch/sl9623/container/duel_tracker/Duel_GT_TS67/Duel_label_AF24/20250123_10_10min_dueling_label.csv"),
        ("/scratch/sl9623/container/prediction/h5_files/filtered_TS67/20250127/basler_record_20250127_16_filtered.h5",
         "/scratch/sl9623/container/duel_tracker/Duel_GT_TS67/Duel_label_AF24/20250127_16_10min_dueling_label.csv"),
        ("/scratch/sl9623/container/prediction/h5_files/filtered_TS67/20250128/basler_record_20250128_07_filtered.h5",
         "/scratch/sl9623/container/duel_tracker/Duel_GT_TS67/Duel_label_AF24/20250128_07_10min_dueling_label.csv"),
        ("/scratch/sl9623/container/prediction/h5_files/filtered_TS67/20250128/basler_record_20250128_12_filtered.h5",
         "/scratch/sl9623/container/duel_tracker/Duel_GT_TS67/Duel_label_AF24/20250128_12_10min_dueling_label.csv"),
        ("/scratch/sl9623/container/prediction/h5_files/filtered_TS67/20250203/basler_record_20250203_18_filtered.h5",
         "/scratch/sl9623/container/duel_tracker/Duel_GT_TS67/Duel_label_AF24/20250203_18_10min_dueling_label.csv"),
        ("/scratch/sl9623/container/prediction/h5_files/filtered_TS67/20250209/basler_record_20250209_17_filtered.h5",
         "/scratch/sl9623/container/duel_tracker/Duel_GT_TS67/Duel_label_AF24/20250209_17_10min_dueling_label.csv")
    ]
    
    test_24AF = [
        ("/scratch/sl9623/container/prediction/h5_files/filtered_TS67/20250203/basler_record_20250203_09_filtered.h5",
         "/scratch/sl9623/container/duel_tracker/Duel_GT_TS67/Duel_label_AF24/20250203_09_10min_dueling_label.csv")
    ]

    train_alltime = [        
        ("/scratch/sl9623/container/prediction/h5_files/filtered_TS67/20250120/basler_record_20250120_07_filtered.h5",
         "/scratch/sl9623/container/duel_tracker/Duel_GT_TS67/Duel_label_AF24_plus/20250120_07_10min_dueling_label.csv"),
        ("/scratch/sl9623/container/prediction/h5_files/filtered_TS67/20250121/basler_record_20250121_12_filtered.h5",
         "/scratch/sl9623/container/duel_tracker/Duel_GT_TS67/Duel_label_AF24_plus/20250121_12_10min_dueling_label.csv"),
        ("/scratch/sl9623/container/prediction/h5_files/filtered_TS67/20250123/basler_record_20250123_12_filtered.h5",
         "/scratch/sl9623/container/duel_tracker/Duel_GT_TS67/Duel_label_AF24_plus/20250123_12_10min_dueling_label.csv"),
        ("/scratch/sl9623/container/prediction/h5_files/filtered_TS67/20250126/basler_record_20250126_12_filtered.h5",
         "/scratch/sl9623/container/duel_tracker/Duel_GT_TS67/Duel_label_AF24_plus/20250126_12_10min_dueling_label.csv"),
        ("/scratch/sl9623/container/prediction/h5_files/filtered_TS67/20250127/basler_record_20250127_12_filtered.h5",
         "/scratch/sl9623/container/duel_tracker/Duel_GT_TS67/Duel_label_AF24_plus/20250127_12_10min_dueling_label.csv"),
        ("/scratch/sl9623/container/prediction/h5_files/filtered_TS67/20250207/basler_record_20250207_12_filtered.h5",
         "/scratch/sl9623/container/duel_tracker/Duel_GT_TS67/Duel_label_AF24_plus/20250207_12_10min_dueling_label.csv"),
        ("/scratch/sl9623/container/prediction/h5_files/filtered_TS67/20250213/basler_record_20250213_11_filtered.h5",
         "/scratch/sl9623/container/duel_tracker/Duel_GT_TS67/Duel_label_AF24_plus/20250213_11_10min_dueling_label.csv"),
        ("/scratch/sl9623/container/prediction/h5_files/filtered_TS67/20250121/basler_record_20250121_16_filtered.h5",
         "/scratch/sl9623/container/duel_tracker/Duel_GT_TS67/Duel_label_AF24_plus/20250121_16_10min_dueling_label.csv"),
        ("/scratch/sl9623/container/prediction/h5_files/filtered_TS67/20250122/basler_record_20250122_16_filtered.h5",
         "/scratch/sl9623/container/duel_tracker/Duel_GT_TS67/Duel_label_AF24/20250122_16_10min_dueling_label.csv"),
        ("/scratch/sl9623/container/prediction/h5_files/filtered_TS67/20250123/basler_record_20250123_10_filtered.h5",
         "/scratch/sl9623/container/duel_tracker/Duel_GT_TS67/Duel_label_AF24/20250123_10_10min_dueling_label.csv"),
        ("/scratch/sl9623/container/prediction/h5_files/filtered_TS67/20250127/basler_record_20250127_16_filtered.h5",
         "/scratch/sl9623/container/duel_tracker/Duel_GT_TS67/Duel_label_AF24/20250127_16_10min_dueling_label.csv"),
        ("/scratch/sl9623/container/prediction/h5_files/filtered_TS67/20250128/basler_record_20250128_07_filtered.h5",
         "/scratch/sl9623/container/duel_tracker/Duel_GT_TS67/Duel_label_AF24/20250128_07_10min_dueling_label.csv"),
        ("/scratch/sl9623/container/prediction/h5_files/filtered_TS67/20250128/basler_record_20250128_12_filtered.h5",
         "/scratch/sl9623/container/duel_tracker/Duel_GT_TS67/Duel_label_AF24/20250128_12_10min_dueling_label.csv"),
        ("/scratch/sl9623/container/prediction/h5_files/filtered_TS67/20250209/basler_record_20250209_17_filtered.h5",
         "/scratch/sl9623/container/duel_tracker/Duel_GT_TS67/Duel_label_AF24/20250209_17_10min_dueling_label.csv"),
    ]
    
    test_file_pairs = [
        ("/scratch/sl9623/container/prediction/h5_files/filtered_TS67/20250203/basler_record_20250203_09_filtered.h5",
         "/scratch/sl9623/container/duel_tracker/Duel_GT_TS67/feeding_timespan/duel_label_20250203_09_AF1.csv"),
        ("/scratch/sl9623/container/prediction/h5_files/filtered_TS67/20250203/basler_record_20250203_13_filtered.h5",
         "/scratch/sl9623/container/duel_tracker/Duel_GT_TS67/feeding_timespan/duel_label_20250203_13_AF5.csv"),
        ("/scratch/sl9623/container/prediction/h5_files/filtered_TS67/20250203/basler_record_20250203_18_filtered.h5",
         "/scratch/sl9623/container/duel_tracker/Duel_GT_TS67/feeding_timespan/duel_label_20250203_18_AF10.csv"),
        ("/scratch/sl9623/container/prediction/h5_files/filtered_TS67/20250204/basler_record_20250204_07_filtered.h5",
         "/scratch/sl9623/container/duel_tracker/Duel_GT_TS67/feeding_timespan/duel_label_20250204_07_AF23.csv"),
        ("/scratch/sl9623/container/prediction/h5_files/filtered_TS67/20250204/basler_record_20250204_12_filtered.h5",
         "/scratch/sl9623/container/duel_tracker/Duel_GT_TS67/feeding_timespan/duel_label_20250204_12_AF28.csv"),
        ("/scratch/sl9623/container/prediction/h5_files/filtered_TS67/20250204/basler_record_20250204_17_filtered.h5",
         "/scratch/sl9623/container/duel_tracker/Duel_GT_TS67/feeding_timespan/duel_label_20250204_17_AF33.csv"),
    ]
    
    # === Output Model Directory ===
    model_dir = "/scratch/sl9623/container/duel_tracker/TS67_alltime_0612"

    # === Run Full Pipeline: Training + Evaluation ===
    #main_train_and_evaluate(train_alltime, test_24AF, model_dir)
    evaluate_and_plot_stgcn(train_alltime, test_file_pairs, model_dir)