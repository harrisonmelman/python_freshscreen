"""Pick a default intensity window by sampling an n5 volume."""
import re

import numpy as np
import zarr

from .config import N5_SUFFIX


def _pick_array(node, max_voxels):
    """Return an array to sample: the node itself, or the best pyramid level in a group."""
    if isinstance(node, zarr.Array):
        return node
    levels = sorted(
        (k for k in node.array_keys() if re.fullmatch(r"s\d+", k)),
        key=lambda k: int(k[1:]),
    )
    if not levels:
        raise ValueError(f"No array or s0/s1/... pyramid levels found in {node.name!r}")
    # Finest level that fits the voxel budget; otherwise the coarsest available
    for k in levels:
        if node[k].size <= max_voxels:
            return node[k]
    return node[levels[-1]]


def compute_display_range(n5_path, max_voxels=2000000, lower_pct=0.5, upper_pct=99.5, ignore_value=0, seed=0):
    """returns {"range": [lo, hi], "window": [min, max]} for a Neuroglancer invlerp control

    range is the percentile window shown on screen; window is the extent of the slider."""
    root = zarr.open(zarr.N5Store(n5_path), mode="r")
    node = root[N5_SUFFIX]
    arr = _pick_array(node, max_voxels)
    data = np.asarray(arr[:]).ravel()

    if np.issubdtype(data.dtype, np.floating):
        data = data[np.isfinite(data)]
    if ignore_value is not None:
        data = data[data != ignore_value]

    if data.size == 0:
        # Nothing usable sampled: fall back to the dtype's natural range
        if np.issubdtype(arr.dtype, np.integer):
            info = np.iinfo(arr.dtype)
            return {"range": [float(info.min), float(info.max)],
                    "window": [float(info.min), float(info.max)]}
        return {"range": [0.0, 1.0], "window": [0.0, 1.0]}

    lo, hi = np.percentile(data, [lower_pct, upper_pct])
    if hi <= lo:  # constant data
        hi = lo + 1
    return {
        "range": [float(lo), float(hi)],
        "window": [float(min(data.min(), lo)), float(max(data.max(), hi))],
    }
