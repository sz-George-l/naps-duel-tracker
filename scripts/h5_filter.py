"""Read in the h5 and display basic info
"""

import h5py
import numpy as np
import copy
import scipy.stats as stats
import pandas as pd
import scipy.ndimage
import argparse
import os


def standardize_track_names(track_names):
    """Standardize track names by removing byte format, stripping 'ArUcoTag', and fixing #3.0 to #3."""
    def clean_track(track):
        track = track.decode('utf-8') if isinstance(track, bytes) else str(track)
        track = track.replace("ArUcoTag", "").strip()
        if track.endswith(".0"):
            track = track[:-2]  # Remove the final '.0'
        return track
    
    cleaned_tracks = [clean_track(track) for track in track_names]
    return cleaned_tracks

def diff(node_loc, diff_func=np.gradient, **kwargs):
    """
    node_loc is a [frames, 2] arrayF

    win defines the window to smooth over

    poly defines the order of the polynomial
    to fit with

    """
    node_loc_vel = np.zeros_like(node_loc)
    for c in range(node_loc.shape[-1]):
        node_loc_vel[:, c] = diff_func(node_loc[:, c], **kwargs)

    node_vel = np.linalg.norm(node_loc_vel, axis=1)

    return node_vel

def flatten_features(x, axis=0):

    if axis != 0:
        # Move time axis to the first dim
        x = np.moveaxis(x, axis, 0)

    # Flatten to 2D.
    initial_shape = x.shape
    x = x.reshape(len(x), -1)

    return x, initial_shape

def unflatten_features(x, initial_shape, axis=0):
    # Reshape.
    x = x.reshape(initial_shape)

    if axis != 0:
        # Move time axis back
        x = np.moveaxis(x, 0, axis)

    return x

def smooth_median(x, window=5, axis=0, inplace=False):
    if axis != 0 or x.ndim > 1:
        if not inplace:
            x = x.copy()

        # Reshape to (time, D)
        x, initial_shape = flatten_features(x, axis=axis)

        # Apply function to each slice
        for i in range(x.shape[1]):
            x[:, i] = smooth_median(x[:, i], window, axis=0)

        # Restore to original shape
        x = unflatten_features(x, initial_shape, axis=axis)
        return x

    y = scipy.signal.medfilt(x.copy(), window)
    y = y.reshape(x.shape)
    mask = np.isnan(y) & (~np.isnan(x))
    y[mask] = x[mask]
    return y

def fill_missing(x, kind="nearest", axis=0, **kwargs):
    """Fill missing values in a timeseries.

    Args:
        x: Timeseries of shape (time, ...) or with time axis specified by axis.
        kind: Type of interpolation to use. Defaults to "nearest".
        axis: Time axis (default: 0).

    Returns:
        Timeseries of the same shape as the input with NaNs filled in.

    Notes:
        This uses pandas.DataFrame.interpolate and accepts the same kwargs.
    """
    if x.ndim > 2:
        # Reshape to (time, D)
        x, initial_shape = flatten_features(x, axis=axis)

        # Interpolate.
        x = fill_missing(x, kind=kind, axis=0, **kwargs)

        # Restore to original shape
        x = unflatten_features(x, initial_shape, axis=axis)

        return x
    return pd.DataFrame(x).interpolate(method=kind, axis=axis, **kwargs).to_numpy()

def instance_node_velocities(fly_node_locations, start_frame, end_frame):
    frame_count = len(range(start_frame, end_frame))
    if len(fly_node_locations.shape) == 4:
        fly_node_velocities = np.zeros(
            (frame_count, fly_node_locations.shape[1], fly_node_locations.shape[3])
        )
        for fly_idx in range(fly_node_locations.shape[3]):
            for n in range(0, fly_node_locations.shape[1]):
                fly_node_velocities[:, n, fly_idx] = diff(
                    fly_node_locations[start_frame:end_frame, n, :, fly_idx]
                )
    else:
        fly_node_velocities = np.zeros((frame_count, fly_node_locations.shape[1]))
        for n in range(0, fly_node_locations.shape[1] - 1):
            fly_node_velocities[:, n] = diff(
                fly_node_locations[start_frame:end_frame, n, :]
            )

    return fly_node_velocities

def smooth_gaussian(x, std=1, window=5, axis=0, inplace=False):
    if axis != 0 or x.ndim > 1:
        if not inplace:
            x = x.copy()

        # Reshape to (time, D)
        x, initial_shape = flatten_features(x, axis=axis)

        # Apply function to each slice
        for i in range(x.shape[1]):
            x[:, i] = smooth_gaussian(x[:, i], std, window, axis=0)

        # Restore to original shape
        x = unflatten_features(x, initial_shape, axis=axis)
        return x

    y = (
        pd.DataFrame(x.copy())
        .rolling(window, win_type="gaussian", center=True)
        .mean(std=std)
        .to_numpy()
    )
    y = y.reshape(x.shape)
    mask = np.isnan(y) & (~np.isnan(x))
    y[mask] = x[mask]
    return y
def velfilt(locs, thresh, antenna_tip_nodes=None, antenna_tip_thresh=None):
    """
    Velocity-based filtering with optional special thresholds for antenna tip nodes.

    Args:
        locs (np.ndarray): shape [T, N, 2, F]
        thresh (float): default velocity threshold for all nodes
        antenna_tip_nodes (list[int] or set[int], optional): node indices to treat specially
        antenna_tip_thresh (float, optional): custom threshold for antenna tip nodes

    Returns:
        velsbool (np.ndarray): shape [T-1, N, F], True where velocity exceeds threshold
    """
    filledlocs = fill_missing(locs[:, :, :, :], kind="pad")
    vels = instance_node_velocities(filledlocs, 1, locs.shape[0])  # shape [T-1, N, F]

    if antenna_tip_nodes is None or antenna_tip_thresh is None:
        # Default behavior for all nodes
        return vels > thresh

    # Start with default threshold mask
    velsbool = vels > thresh

    # Override for antenna tip nodes
    for node_idx in antenna_tip_nodes:
        velsbool[:, node_idx, :] = vels[:, node_idx, :] > antenna_tip_thresh

    return velsbool


def coordfilt(velsbool, limit):
    sums = np.sum(velsbool, axis=1)
    coord1dbool = sums >= limit
    num_nodes = velsbool.shape[1]
    coordbool = np.tile(coord1dbool[:, np.newaxis, :], (1, num_nodes, 1))
    return coordbool

def integratedvelfilter(locations, velsbool, extvelsbool, coordbool):
    uncoordvelsbool = velsbool & ~coordbool
    finalbool = uncoordvelsbool + extvelsbool
    coordsaved = np.sum(velsbool) - np.sum(uncoordvelsbool) 
    print("Vel Outliers Saved by Coord: ", coordsaved)
    final2dbool = np.stack((finalbool,finalbool), axis=2)
    print("Final Filter: ", np.sum(finalbool))
    locationsfilter = copy.deepcopy(locations) 
    locationsfilter = locationsfilter[1:,:,:,:]
    previousnans = np.isnan(locationsfilter[:,:,0,:])
    previousandpresent = previousnans & finalbool
    locationsfilter[final2dbool] = np.nan
    return(locationsfilter)

def shrink_locs(locations, node, track1, track2):
    filt1 = ~np.isnan(locations[:,node_names.index(node),0,track1])
    filt2 = ~np.isnan(locations[:,node_names.index(node),0,track2])
    filt = filt1 & filt2
    array1 = locations[filt, node_names.index(node), :, track1]
    array2 = locations[filt, node_names.index(node), :, track2]
    distances = np.linalg.norm(array1 - array2, axis=1)
    return distances

def removelargeedges (locations, frame_start, frame_end, zscore_threshold = 5):
    for track_pos in range(locations.shape[3]):
        for edge_nodeA in range(locations.shape[1]):
            for edge_nodeB in range(locations.shape[1]):
                if edge_nodeA < edge_nodeB:
                    array1 = locations[frame_start:frame_end, edge_nodeA, :, track_pos]
                    array2 = locations[frame_start:frame_end, edge_nodeB, :, track_pos]
                    distances = np.linalg.norm(array1-array2, axis=1)
                    zscore_array = stats.zscore(distances, nan_policy = 'omit')
                    locations[frame_start:frame_end, edge_nodeA, :, track_pos][zscore_array > zscore_threshold] = np.nan
                    locations[frame_start:frame_end, edge_nodeB, :, track_pos][zscore_array > zscore_threshold] = np.nan
    return locations

def removeSharedNodes (locations, frame_start, frame_end, shared_dist_threshold = 50):    
    for track_posA in range(locations.shape[3]):        
        for track_posB in range(locations.shape[3]):
            for nodeA in range(locations.shape[1]):
                for nodeB in range(locations.shape[1]):
                    if track_posA < track_posB:
                        array1 = locations[frame_start:frame_end, nodeA, :, track_posA]
                        array2 = locations[frame_start:frame_end, nodeB, :, track_posB]
                        distance_array = np.linalg.norm(array1-array2, axis=1)
                        locations[frame_start:frame_end,nodeA,:,track_posA][distance_array < shared_dist_threshold] = np.nan
                        locations[frame_start:frame_end,nodeB,:,track_posB][distance_array < shared_dist_threshold] = np.nan
    return locations

def adaptive_fill(data, kind="linear", window=10):
    assert data.ndim == 4 and data.shape[2] == 2, "Expected shape [F, N, 2, T]"
    filled = data.copy()
    F, N, _, T = filled.shape
    long_gap_filled = 0
    for t in range(T):  # per timepoint
        for n in range(N):  # per node
            coords = filled[:, n, :, t]  # shape [F, 2]
            for dim in range(2):  # x or y
                series = pd.Series(coords[:, dim])
                # First attempt: interpolate only gaps ≤ window using fill_missing with limit
                interpolated = fill_missing(
                    series.to_numpy(), kind=kind, axis=0, limit=window, limit_direction='both').squeeze()
                series = pd.Series(interpolated)
                # Second pass: fill remaining NaNs (large or edge gaps) with nearest available value
                is_nan = series.isna()
                i = 0
                while i < F:
                    if is_nan[i]:
                        start = i
                        while i < F and is_nan[i]:
                            i += 1
                        end = i
                        gap_len = end - start
                        if start > 0:
                            fill_value = series[start - 1]
                        elif end < F:
                            fill_value = series[end]
                        else:
                            continue  # All NaN
                        series[start:end] = fill_value
                        long_gap_filled += gap_len
                    else:
                        i += 1
                coords[:, dim] = series.to_numpy()
            filled[:, n, :, t] = coords
    return filled, long_gap_filled

def process_h5(input_file, output_file):
    with h5py.File(input_file, "r") as f:
        dset_names = list(f.keys())
        locations = f["tracks"][:].T
        node_names = [n.decode() for n in f["node_names"][:]]
        track_names = standardize_track_names(f["track_names"][:])
        point_scores = f["point_scores"][:]
        instance_scores = f["instance_scores"][:]
        track_occupancy = f["track_occupancy"][:]
        tracking_scores = f["tracking_scores"][:]
    
    print("=== Locations data shape ===")
    print(locations.shape)
    print()

    skeletalset = (0, 1, 2, 3, 10, 11, 12, 13, 14, 15, 16)
    node_names = node_names[0:4] + node_names[10:17]
    locations = locations[:, skeletalset, :, :]
    print(node_names)
    
    sum_of_defined_nodes = np.apply_over_axes(np.sum, ~np.isnan(locations[:, :, 0, :]), [0, 2])
    velparam, coordparam, edgeparam, spaceparam, interpparam = 10, 2, 5, 1, 10
    print("Prefilter: ", np.sum(sum_of_defined_nodes[:]))
    
    velsbool = velfilt(locations, velparam)
    coordbool = coordfilt(velsbool, coordparam)
    print("Vels Identified: ", np.sum(velsbool))
    print("Coord Identified: ", np.sum(coordbool))
    
    extvelsbool = velfilt(locations, 100, antenna_tip_nodes=[4, 6], antenna_tip_thresh=200)
    print("Extreme Vels Filtered: ", np.sum(extvelsbool))
    
    integfilter = integratedvelfilter(locations, velsbool, extvelsbool, coordbool)
    sum_of_defined_filtered_nodes = np.apply_over_axes(np.sum, ~np.isnan(integfilter[:, :, 0, :]), [0, 2])
    print("PostVelFilter:", np.sum(sum_of_defined_filtered_nodes[:]))
    
    edgefilterlocs = removelargeedges(integfilter, 0, 108000, zscore_threshold=edgeparam)
    sum_of_defined_filtered_nodes = np.apply_over_axes(np.sum, ~np.isnan(edgefilterlocs[:, :, 0, :]), [0, 2])
    print("PostEdgeFilter:", np.sum(sum_of_defined_filtered_nodes[:]))
    
    spacefilterlocs = removeSharedNodes(edgefilterlocs, 0, 108000, shared_dist_threshold=spaceparam)
    sum_of_defined_filtered_nodes = np.apply_over_axes(np.sum, ~np.isnan(spacefilterlocs[:, :, 0, :]), [0, 2])
    print("PostSpaceFilter:", np.sum(sum_of_defined_filtered_nodes[:]))
    
    interpdlocs, long_gap_filled = adaptive_fill(spacefilterlocs, window=interpparam)
    sum_of_defined_interp_nodes = np.apply_over_axes(np.sum, ~np.isnan(interpdlocs[:, :, 0, :]), [0, 2])
    print("PostInterp:", np.sum(sum_of_defined_interp_nodes[:]))
    
    nan_count_after = np.isnan(interpdlocs).sum()
    print(f"Filled Long Gap NANs: {long_gap_filled}")
    print(f"Remaining NANs: {nan_count_after}")
    
    tracks = interpdlocs.T
    
    with h5py.File(output_file, "w") as f:
        f.create_dataset("instance_scores", data=instance_scores)
        f.create_dataset("node_names", data=node_names)
        f.create_dataset("point_scores", data=point_scores)
        f.create_dataset("track_names", data=track_names)
        f.create_dataset("track_occupancy", data=track_occupancy)
        f.create_dataset("tracking_scores", data=tracking_scores)
        f.create_dataset("tracks", data=tracks)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an HDF5 file and output a filtered version.")
    parser.add_argument("input_file", type=str, help="Path to the input HDF5 file")
    parser.add_argument("output_file", type=str, help="Path to the output HDF5 file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        exit(1)
    
    process_h5(args.input_file, args.output_file)
