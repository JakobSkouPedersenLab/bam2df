from bisect import bisect_right
from collections import Counter
from math import log2
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import pysam
import os

from src.features.filters import _read_filters
from scipy.sparse import csr_matrix

import pyBigWig

# Caching setup: dictionary to store coverage arrays for regions
coverage_cache: dict[tuple[str, int], list[int]] = ({})  # Key: (contig, segment_start); Value: coverage array for that segment
CACHE_SEGMENT_SIZE = 10_000 # e.g., cache coverage in 1000-base segments to reuse across close positions
MAX_SEGMENTS_CACHED = 20  # Limit cache size to avoid memory blow-up in extremely deep coverage scenarios
current_cache_contig = None  # Track current contig in cache for easy invalidation
bamfile_cache = {}  # Cache for opened BAM files

from dataclasses import dataclass, field, replace
import polars as pl
from pathlib import Path

@dataclass
class Read2VecConfig:
    interim_dir: Path
    labels_csv: Path
    processed_dir: Path
    test_split: float = 0.2
    seed: int = 0

    # bag structure
    micro_bp: int = 2_000        # e.g. 2kb micro bins
    window_bp: int = 200_000     # e.g. 200kb windows (if not using bin_idx)
    use_bin_idx_as_window: bool = True

    # preprocessing
    compute_scalers: bool = True
    scaler_max_samples: int = 200     # cap samples used for global stats
    scaler_max_rows_per_sample: int = 200_000
    
    # features
    num_features: list[str] = field(default_factory=list)
    cat_features: list[str] = field(default_factory=list)

@dataclass
class ChromStats:
    chrom: str = None
    n_reads: int = 0
    n_mismatches: int = 0
    n_matches: int = 0
    n_indels: int = 0
    n_total: int = 0
    n_skipped: int = 0
    n_evicted: int = 0
    n_invalid_fragments: int = 0

    def __iadd__(self, other: "ChromStats"):
        self.n_reads += other.n_reads
        self.n_mismatches += other.n_mismatches
        self.n_matches += other.n_matches
        self.n_indels += other.n_indels
        self.n_total += other.n_total
        self.n_skipped += other.n_skipped
        self.n_evicted += other.n_evicted
        self.n_invalid_fragments += other.n_invalid_fragments
        return self

    def __str__(self):
        return (f"[{self.chrom}] reads={self.n_reads:,}, mismatches={self.n_mismatches:,}, "
                f"matches={self.n_matches:,}, indels={self.n_indels:,}, "
                f"opportunities={self.n_total:,}, skipped={self.n_skipped:,}, "
                f"evicted={self.n_evicted:,}, invalid={self.n_invalid_fragments:,}")
            
@dataclass
class GenomeStats:
    """
    Holds aggregated stats for an entire BAM (all chromosomes),
    along with per-chromosome breakdowns.
    """
    per_chrom: Dict[str, ChromStats] = field(default_factory=dict)
    total: ChromStats = field(default_factory=lambda: ChromStats(chrom="total"))

    def add(self, stats: ChromStats):
        """Add per-chromosome stats and update total."""
        self.per_chrom[stats.chrom] = stats
        self.total += stats
        
    def __iadd__(self, stats: ChromStats):
        self.add(stats)
        return self

    def __str__(self):
        lines = [f"[GenomeStats] Combined: {self.total}"]
        for chrom, cs in self.per_chrom.items():
            lines.append(f"  {chrom}: {cs}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to a serializable dict."""
        return {
            "total": self.total.__dict__,
            "per_chrom": {k: v.__dict__ for k, v in self.per_chrom.items()},
        }

@dataclass
class FilterContext:
    """
    Container for all blacklist and coverage-related filtering resources
    for a single patient.
    """
    genomic_blacklist_by_chr: Optional[dict[str, tuple]] = None
    donor_blacklist_by_chr: Optional[dict[str, tuple]] = None
    patient_blacklists_by_chr: Optional[dict[str, dict[str, tuple]]] = None
    # Patient germline calls, matched on exact allele (see src/utility/variant_blacklist.py).
    # chr -> {(start0, ref, alt)}
    snv_blacklist_by_chr: dict[str, set[tuple[int, str, str]]] = field(default_factory=dict)
    # chr -> {(start0, end0, ref, alt)}, VCF-style anchored alleles
    indel_blacklist_by_chr: dict[str, set[tuple[int, int, str, str]]] = field(default_factory=dict)

    def for_chromosome(self, chrom: str) -> "FilterContext":
        """
        Copy holding only `chrom`'s germline calls.

        Each chromosome runs in its own spawned process, which receives a pickled copy of the
        context; sending only one chromosome's calls avoids copying a genome's worth of variants
        into every worker.
        """
        return replace(
            self,
            snv_blacklist_by_chr={chrom: self.snv_blacklist_by_chr[chrom]}
            if chrom in self.snv_blacklist_by_chr else {},
            indel_blacklist_by_chr={chrom: self.indel_blacklist_by_chr[chrom]}
            if chrom in self.indel_blacklist_by_chr else {},
        )

@dataclass
class MicrosatelliteLookup:
    starts: list[int]
    ends: list[int]
    motifs: list[str]
    def query(self, pos0: int) -> tuple[bool, Optional[str]]:
        i = bisect_right(self.starts, pos0) - 1
        if i >= 0 and pos0 < self.ends[i]:
            return True, self.motifs[i]
        return False, None

@dataclass
class RepeatMaskerLookup:
    starts: list[int]
    ends: list[int]
    rep_classes: list[str]
    def query(self, pos0: int) -> tuple[bool, Optional[str]]:
        i = bisect_right(self.starts, pos0) - 1
        if i >= 0 and pos0 < self.ends[i]:
            return True, self.rep_classes[i]
        return False, None

class ReferenceCache:
    """Fetch reference chunks and keep a sliding window cache per contig."""

    def __init__(self, fasta_path: str, window: int = 1_000_000):
        self.fasta = pysam.FastaFile(fasta_path)
        self.window = window
        self._cache: Dict[str, Tuple[int, int, str]] = {}

    def fetch(self, contig: str, pos: int, flank: int) -> str:
        """Return *2 × flank + 1* bases centred on 1-based **pos**."""
        start = max(pos - flank - 1, 0)
        end = pos + flank
        lo, hi, seq = self._cache.get(contig, (None, None, None))
        if seq and lo is not None and hi is not None and lo <= start and end <= hi:
            off = start - lo
            return seq[off : off + (2 * flank + 1)]
        new_lo = max(0, pos - self.window // 2)
        new_hi = new_lo + self.window
        seq = self.fasta.fetch(contig, new_lo, new_hi)
        self._cache[contig] = (new_lo, new_hi, seq)
        off = start - new_lo
        return seq[off : off + (2 * flank + 1)]
    
def calculate_local_complexity(seq: str) -> tuple[float, float]:
    """Compute entropy-based complexity for k=1,2 in one pass."""
    if not seq:
        return 0.0, 0.0

    n = len(seq)
    c1 = Counter(seq)
    p1 = [v / n for v in c1.values()]
    h1 = -sum(p * log2(p) for p in p1 if p > 0)

    if n > 1:
        c2 = Counter(seq[i:i+2] for i in range(n - 1))
        p2 = [v / (n - 1) for v in c2.values()]
        h2 = -sum(p * log2(p) for p in p2 if p > 0)
    else:
        h2 = 0.0

    return h1, h2

def _bw_value_at_1based(
    bw: Optional["pyBigWig.bigWigFile"],
    chrom: str,
    pos_1based: int,
) -> Optional[float]:
    """
    Return bigWig value at a 1-based genomic position.
    Uses 0-based half-open indexing internally.
    """
    if bw is None:
        return None

    try:
        vals = bw.values(chrom, pos_1based - 1, pos_1based)
    except RuntimeError:
        return None

    if not vals:
        return None

    v = vals[0]
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None

    return float(v)

def contains_indel_or_clip(n_ins, n_del, n_soft, n_hard) -> bool:
    return (n_ins + n_del + n_soft + n_hard) > 0

def count_other_errors_in_read(md_tag: str) -> int:
    """
    Fast mismatch counter for an MD tag.
    Counts alphabetic bases (A/C/G/T) that are not part of deletions (^...) 
    and ignores numeric match runs entirely.
    """
    if not md_tag:
        return 0

    mismatch_count = 0
    i = 0
    n = len(md_tag)

    while i < n:
        c = md_tag[i]

        if c == "^":
            # Skip deletion block (e.g. ^ACGT)
            i += 1
            while i < n and md_tag[i].isalpha():
                i += 1
            continue

        if c.isdigit():
            # Skip entire numeric block (e.g. 23)
            while i < n and md_tag[i].isdigit():
                i += 1
            continue

        # Count as mismatch (A/C/G/T)
        if c.isalpha():
            mismatch_count += 1

        i += 1

    return mismatch_count

def get_bam_handle(bam_path: str) -> pysam.AlignmentFile:
    """Return a cached pysam.AlignmentFile handle or open one if not already cached."""
    if bam_path not in bamfile_cache:
        if not os.path.exists(bam_path):
            raise FileNotFoundError(f"BAM file not found: {bam_path}")
        # Use a small number of threads; pysam threads control decompression, not compute
        bamfile_cache[bam_path] = pysam.AlignmentFile(bam_path, "rb", threads=2)
    return bamfile_cache[bam_path]

def _get_read_depth(bam_path: str,
                    contig: str,
                    position: int,
                    min_mapq: int,
                    min_frag_len: int,
                    max_frag_len: int) -> int:
    """
    Return the read depth at a specific 0-based reference position on a contig.

    This function uses a global LRU-style cache of fixed-size coverage segments
    to avoid repeated pysam.count_coverage() calls for nearby positions.

    Parameters
    ----------
    bam_path : str
        Path to BAM file (will be cached and reused).
    contig : str
        Chromosome / contig name.
    position : int
        0-based genomic coordinate.

    Returns
    -------
    int
        Total read depth at the requested position.
    """
    global current_cache_contig, coverage_cache
    bam_file = get_bam_handle(bam_path)

    # If we switched to a new contig, clear old coverage segments
    if contig != current_cache_contig:
        coverage_cache.clear()
        current_cache_contig = contig

    # Compute which segment this position belongs to
    segment_start = (position // CACHE_SEGMENT_SIZE) * CACHE_SEGMENT_SIZE
    segment_key = (contig, segment_start, min_mapq, min_frag_len, max_frag_len)

    # Try to reuse cached coverage for this segment
    segment_depth = coverage_cache.get(segment_key)
    if segment_depth is None:
        segment_end = segment_start + CACHE_SEGMENT_SIZE
        
        # Create a callback that applies the same filters with the correct parameters
        def filter_callback(read):
            return _read_filters(read=read,
                                 min_mapq=min_mapq,
                                 min_frag_len=min_frag_len,
                                 max_frag_len=max_frag_len)

        # Compute depth arrays for this chunk
        A, C, G, T = bam_file.count_coverage(
            contig,
            start=segment_start,
            stop=segment_end,
            read_callback=filter_callback
        )
        # Combine base-specific counts into total depth
        segment_depth = [a + c + g + t for a, c, g, t in zip(A, C, G, T)]
        coverage_cache[segment_key] = segment_depth

        # Simple cache eviction policy: drop oldest segment if over limit
        if len(coverage_cache) > MAX_SEGMENTS_CACHED:
            oldest_key = next(iter(coverage_cache))
            coverage_cache.pop(oldest_key, None)

    # Compute index within this segment and return cached depth
    idx = position - segment_start
    if 0 <= idx < len(segment_depth):
        return segment_depth[idx]
    else:
        # Out of bounds (e.g., end of contig)
        return 0

def rows_to_polars(rows, schema=None) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=schema)

    first_row = rows[0]

    if schema is not None:
        cols = list(schema.keys()) if isinstance(schema, dict) else [f.name for f in schema]
        if isinstance(first_row, dict):
            rows_dicts = [{k: row.get(k, None) for k in cols} for row in rows]
        else:
            rows_dicts = [{k: getattr(row, k, None) for k in cols} for row in rows]
        return pl.DataFrame(rows_dicts, schema=schema)

def get_open_indices(mask: csr_matrix) -> np.ndarray:
    r, c = mask.nonzero()

    # If it's a vector stored as (n,1) -> use row indices
    if mask.shape[1] == 1:
        idx = r
    # If it's (1,n) -> use col indices
    elif mask.shape[0] == 1:
        idx = c
    else:
        # If truly 2D (unexpected), you must define what "open bin" means.
        # This fallback treats row index as bin index.
        idx = r

    idx = np.asarray(idx, dtype=np.int64)
    idx.sort()
    return idx