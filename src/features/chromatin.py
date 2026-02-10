from functools import lru_cache
import numpy as np
import pandas as pd
import scipy as sp
from scipy.signal import convolve
import polars as pl

MAX_DIST_BP = 65535  # UInt16 max

def extract_chromatin_features(
    bins: np.ndarray,
    open_idx: np.ndarray,
    modality: str,
    bin_size_bp: int,
):
    open_bin = is_open_bin(bins, open_idx).astype(np.float32)
    dist_bins = nearest_distance_in_bins(bins, open_idx)
    dist_to_open_bp = np.minimum(dist_bins * bin_size_bp, MAX_DIST_BP).astype(np.uint16)
    
    prefix = modality.lower()
    return {
        f"{prefix}_open": open_bin,
        f"{prefix}_dist_to_open": dist_to_open_bp.astype(np.float32),
    }

def is_open_bin(bins: np.ndarray, open_idx: np.ndarray) -> np.ndarray:
    if open_idx.size == 0:
        return np.zeros(bins.shape[0], dtype=np.float32)
    j = np.searchsorted(open_idx, bins, side="left")
    j = np.clip(j, 0, open_idx.size - 1)
    return (open_idx[j] == bins).astype(np.float32)

def nearest_distance_in_bins(bins: np.ndarray, open_idx: np.ndarray) -> np.ndarray:
    """
    bins: int array of shape (n,)
    open_idx: sorted int array of open bin indices
    returns: float32 array of shape (n,) distances in bins (not bp)
    """
    n = bins.shape[0]
    out = np.full(n, np.nan, dtype=np.float32)
    if open_idx.size == 0:
        return out

    # insertion positions
    j = np.searchsorted(open_idx, bins, side="left")

    # candidate neighbors
    left = np.clip(j - 1, 0, open_idx.size - 1)
    right = np.clip(j, 0, open_idx.size - 1)

    d_left = np.abs(bins - open_idx[left])
    d_right = np.abs(bins - open_idx[right])
    out[:] = np.minimum(d_left, d_right).astype(np.float32)
    return out