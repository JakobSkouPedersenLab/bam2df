from functools import lru_cache
import numpy as np
import pandas as pd
import scipy as sp
from scipy.signal import convolve
import polars as pl

from src.utility.data import ChromatinResources

def extract_chromatin_features(df_chr: pl.DataFrame,
                               chrom: str,
                               chromatin_resources: ChromatinResources):
    """
    Extract per-chromosome consensus chromatin accessibility
    features derived from ATAC-seq and DNase-seq.
    """
    atac = extract_consensus_accessibility_features_chr(
        df_chr,
        chrom,
        chromatin_resources,
        modality="ATAC",
    )
    dnase = extract_consensus_accessibility_features_chr(
        df_chr,
        chrom,
        chromatin_resources,
        modality="DNase",
    )
    
    features: dict[str, np.ndarray] = {}

    for feat_df in (atac, dnase):
        for col in feat_df.columns:
            features[col] = feat_df[col].to_numpy()

    return features

def extract_consensus_accessibility_features_chr(
    df_chr: pl.DataFrame,
    chrom: str,
    chromatin_resources,
    modality="ATAC",
    window_bins=10,
    density_bins=50,
):
    """
    Compute per-row chromatin accessibility features for a single chromosome
    and assay modality.

    This function operates on a Polars DataFrame containing rows from a
    *single chromosome* and a precomputed integer `bin_idx` column. Features
    are derived from a chromosome-wide consensus accessibility mask stored
    in `chromatin_resources`.

    Features computed (per row):
      - <MOD>_consensus:
            Binary indicator of direct overlap with an open chromatin bin.
      - <MOD>_consensus_win{window_bins}:
            Local accessibility in a ±window_bins bin window
            (uniformly smoothed).
      - <MOD>_consensus_density{density_bins}:
            Regional accessibility density in a ±density_bins bin window.
      - <MOD>_dist_to_open_bp:
            Distance (in base pairs) to the nearest open chromatin bin.
    """
    n = len(df_chr)

    open_bin = np.zeros(n, dtype=np.float32)
    open_win = np.zeros(n, dtype=np.float32)
    open_density = np.zeros(n, dtype=np.float32)
    dist_to_open = np.full(n, np.nan, dtype=np.float32)

    win_kernel = np.ones(2 * window_bins + 1, dtype=np.float32)
    win_kernel /= win_kernel.sum()

    dens_kernel = np.ones(2 * density_bins + 1, dtype=np.float32)
    dens_kernel /= dens_kernel.sum()

    mask = chromatin_resources.consensus_masks[modality][chrom]
    mask_dense = mask.toarray().ravel()

    smoothed_win = convolve(mask_dense, win_kernel, mode="same")
    smoothed_dens = convolve(mask_dense, dens_kernel, mode="same")

    open_indices = np.flatnonzero(mask_dense > 0)

    bins = df_chr["bin_idx"].to_numpy()
    valid = bins < mask_dense.shape[0]

    open_bin[valid] = mask_dense[bins[valid]]
    open_win[valid] = smoothed_win[bins[valid]]
    open_density[valid] = smoothed_dens[bins[valid]]

    if len(open_indices) > 0:
        dist = np.abs(
            bins[valid, None] - open_indices[None, :]
        ).min(axis=1)
        dist_to_open[valid] = dist * 10  # bins -> bp

    prefix = modality.lower()

    return pl.DataFrame(
        {
            f"{prefix}_consensus": pl.Series(open_bin),
            f"{prefix}_consensus_win{window_bins}": pl.Series(open_win),
            f"{prefix}_consensus_density{density_bins}": pl.Series(open_density),
            f"{prefix}_dist_to_open_bp": pl.Series(dist_to_open),
        }
    )