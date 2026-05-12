import atexit
import gzip
import pathlib
import shutil
from typing import List, Optional
import numpy as np
import pandas as pd
import polars as pl
import os
import glob

from scipy.sparse import csr_matrix, load_npz

from src.features.constants import _VALID_CHROMS
from src.utility.sample import get_patient_id
from src.utility.data import MicrosatelliteLookup, RepeatMaskerLookup

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

def load_problematic_blacklist(file_path: str, patient_id: str):
    """
    Load the problematic blacklist file for a given patient.
    `file_path` is expected to be the full path to the blacklist file.
    """
    if not file_path or not os.path.exists(file_path):
        print(f"[RawDataLoader:WARN] Problematic blacklist missing for {patient_id}", flush=True)
        return None

    df = pl.read_csv(file_path, separator="\t", has_header=False)
    df.columns = ["chr", "start", "end"]
    return df

def load_microsatellites_for_chrom(path: str | pathlib.Path, chrom: str) -> Optional[MicrosatelliteLookup]:
    starts: list[int] = []
    ends: list[int] = []
    motifs: list[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            row_chrom = parts[1]
            if row_chrom != chrom:
                continue

            start = int(parts[2])
            end = int(parts[3])
            motif = parts[4]

            starts.append(start)
            ends.append(end)
            motifs.append(motif)

    if not starts:
        return None

    order = sorted(range(len(starts)), key=lambda i: starts[i])
    starts = [starts[i] for i in order]
    ends = [ends[i] for i in order]
    motifs = [motifs[i] for i in order]

    return MicrosatelliteLookup(starts=starts, ends=ends, motifs=motifs)

def load_repeatmasker_for_chrom(
    repeatmasker_path: str | pathlib.Path,
    chrom: str,
) -> Optional[RepeatMaskerLookup]:
    """
    Load only RepeatMasker rows for one chromosome from a UCSC rmsk-style text file.

    Expected columns:
      bin swScore milliDiv milliDel milliIns genoName genoStart genoEnd genoLeft
      strand repName repClass repFamily repStart repEnd repLeft id
    """
    starts: list[int] = []
    ends: list[int] = []
    rep_classes: list[str] = []

    path = pathlib.Path(repeatmasker_path)

    collecting = False
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 13:
                continue

            row_chrom = parts[5]
            if row_chrom == chrom:
                collecting = True
                geno_start = int(parts[6])   # 0-based start
                geno_end = int(parts[7])     # 0-based end-exclusive
                rep_class = parts[11]

                starts.append(geno_start)
                ends.append(geno_end)
                rep_classes.append(rep_class)

            elif collecting:
                # File is usually grouped by chromosome; once we've left the block,
                # stop scanning.
                break

    if not starts:
        return None

    order = sorted(range(len(starts)), key=lambda i: starts[i])
    starts = [starts[i] for i in order]
    ends = [ends[i] for i in order]
    rep_classes = [rep_classes[i] for i in order]

    return RepeatMaskerLookup(
        starts=starts,
        ends=ends,
        rep_classes=rep_classes,
    )

def load_bed_intervals(path: str) -> dict[str, list[tuple[int, int]]]:
    """
    Load a BED file into {chrom: [(start, end), ...]}.
    Assumes BED is 0-based, half-open.
    """
    intervals: dict[str, list[tuple[int, int]]] = {}

    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            fields = line.split("\t")
            if len(fields) < 3:
                raise ValueError(f"BED parse error at line {line_no}: expected >=3 columns")

            chrom = fields[0]
            start = int(fields[1])
            end = int(fields[2])

            if end <= start:
                raise ValueError(
                    f"BED parse error at line {line_no}: end must be > start, got {chrom}:{start}-{end}"
                )

            intervals.setdefault(chrom, []).append((start, end))

    for chrom in intervals:
        intervals[chrom].sort()

    return intervals

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