import atexit
import gzip
import pathlib
import shutil
from typing import List
import numpy as np
import pandas as pd
import polars as pl
import os
import glob

from scipy.sparse import csr_matrix, load_npz

from src.features.constants import _VALID_CHROMS
from src.utility.sample import get_patient_id

def load_bed(path: str, label: str):
    if not path:
        return None
    try:
        df = pl.read_csv(path, separator="\t", has_header=False)
        df.columns = ["chr", "start", "end", *[f"col{i}" for i in range(3, len(df.columns))]]
        return df
    except FileNotFoundError:
        print(f"[RawDataLoader:WARN] {label} BED file not found at {path}, skipping.", flush=True)
        return None
    
def load_table(path: str, label: str):
    if not path:
        return None
    try:
        df = pl.read_csv(path, separator="\t")
        df.columns = ["chr", "start", "end", "base", "count"]
        return df
    except FileNotFoundError:
        print(f"[RawDataLoader:WARN] {label} file not found at {path}, skipping.", flush=True)
        return None

def load_snp_blacklist(path: str, patient_id: str):
    if not os.path.exists(path):
        print(f"[RawDataLoader:WARN] SNP blacklist missing for {patient_id}", flush=True)
        return None
    df = pl.read_csv(path, separator="\t")
    rename_map = {c: c.lower() for c in df.columns}
    df = df.rename(rename_map)
    df = df.rename({"#chr": "chr"})
    return df

def load_lowpower_blacklist(path: str, patient_id: str):
    if not os.path.exists(path):
        print(f"[RawDataLoader:WARN] Low-power blacklist missing for {patient_id}", flush=True)
        return None
    df = pl.read_csv(path, separator="\t", has_header=False)
    df.columns = ["chr", "start", "end"]
    return df

def load_problematic_blacklist(base_path: str, patient_id: str):
    """
    Load the problematic blacklist file for a given patient.
    Expected pattern:
        {patient_id}.problematic.merged.gt<number>X.bg
    Example:
        C05638.problematic.merged.gt192X.bg
    """
    pattern = os.path.join(base_path, patient_id, f"{patient_id}.problematic.merged.gt*X.bg")
    matches = glob.glob(pattern)

    if not matches:
        print(f"[RawDataLoader:WARN] Problematic blacklist missing for {patient_id}", flush=True)
        return None
    if len(matches) > 1:
        print(f"[RawDataLoader:WARN] Multiple blacklist files found for {patient_id}: {matches}", flush=True)

    file_path = matches[0]
    df = pl.read_csv(file_path, separator="\t", has_header=False)
    df.columns = ["chr", "start", "end"]
    return df

def load_coverage_cache(base_path: str, patient_id: str, sample_id: str):
    """
    Fetch the path for BigWig coverage file for a given sample and return the path

    Expected path:
        {base_path}/{patient_id}/{sample_id}_cfdna.aligned.sorted.markdup.coverage.bigWig

    Returns:
        pyBigWig.BigWigFile object (open handle)
    """
    bw_path = os.path.join(
        base_path,
        patient_id,
        f"{sample_id}_cfdna.aligned.sorted.markdup.coverage.bigWig"
    )

    if not os.path.exists(bw_path):
        print(f"[RawDataLoader:WARN] Coverage file not found for {sample_id}: {bw_path}", flush=True)
        return None

    return bw_path

def split_by_chr(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Split large blacklist into per-chromosome DataFrames."""
    if df is None or df.is_empty():
        return {}

    # Filter to valid chromosomes only once
    df = df.filter(pl.col("chr").is_in(list(_VALID_CHROMS)))

    # partition_by is highly optimized in Polars (zero-copy per chrom)
    parts = df.partition_by("chr", as_dict=True)

    # keys are tuples, e.g. ('chr1',), so unpack the first element
    return {k[0]: v for k, v in parts.items()}

def prepare_blacklist_arrays(blacklist_by_chr: dict[str, pl.DataFrame]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Convert each per-chromosome blacklist DataFrame into (starts, ends) NumPy arrays.
    """
    out = {}
    for chrom, df in (blacklist_by_chr or {}).items():
        if not df.is_empty():
            df_sorted = df.sort("start")
            out[chrom] = (
                df_sorted["start"].to_numpy(),
                df_sorted["end"].to_numpy()
            )
    return out

def process_vcf(vcf_path):
    # Extract sample_id from the path
    patient_id = get_patient_id(vcf_path)

    # Count the number of header lines
    with gzip.open(vcf_path, 'rt') as f:
        header_lines = 0
        for line in f:
            if line.startswith('#'):
                header_lines += 1
            else:
                break

    # Read the VCF as a DataFrame, skipping the header lines
    vcf_df = pd.read_csv(vcf_path, sep='\t', comment='#', header=None, skiprows=header_lines)

    # Get column names from the last header line
    with gzip.open(vcf_path, 'rt') as f:
        for line in f:
            if line.startswith('#CHROM'):
                columns = line.strip().lstrip('#').split('\t')
                break
    vcf_df.columns = columns

    # Keep only the columns: CHROM, POS, REF, ALT and rename them
    vcf_df = vcf_df[['CHROM', 'POS', 'REF', 'ALT']]
    vcf_df = vcf_df.rename(columns={'CHROM': 'chr', 'POS': 'genomic_pos', 'REF': 'ref', 'ALT': 'obs'})

    # Add the sample_id column
    vcf_df['patient_id'] = patient_id

    return vcf_df

def get_hpc_tmp_dir(prefix: str) -> pathlib.Path:
    """
    Returns a job-local temporary directory suitable for large intermediate data.
    Tries TMPDIR, then /tmp/$SLURM_JOB_ID, then TMP, then /tmp/<prefix>-<pid>.
    Ensures directory exists and registers cleanup.
    """

    candidates = []

    if "TMPDIR" in os.environ:
        candidates.append(os.environ["TMPDIR"])

    if "SLURM_JOB_ID" in os.environ:
        candidates.append(f"/tmp/{os.environ['SLURM_JOB_ID']}")

    if "TMP" in os.environ:
        candidates.append(os.environ["TMP"])

    candidates.append(f"/tmp/{prefix}-{os.getpid()}")

    for path in candidates:
        if path:
            p = pathlib.Path(path)
            try:
                p.mkdir(parents=True, exist_ok=True)
                if os.access(p, os.W_OK):
                    # cleanup on exit
                    atexit.register(lambda: shutil.rmtree(p, ignore_errors=True))
                    return p
            except Exception:
                continue

    raise RuntimeError("Could not create a writable temporary directory")

def load_chr_mask(root: pathlib.Path, modality: str, chrom: str) -> csr_matrix:
    path = (
        root / "chromatin_masks" / modality
        / "sparse_overlaps_by_chromosome" / "consensus"
        / f"{chrom}.npz"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing chromatin mask: {path}")
    return load_npz(path).tocsr()