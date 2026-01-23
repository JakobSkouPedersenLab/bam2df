import atexit
from collections import Counter
from itertools import product
from math import log2
import os
import pathlib
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import pysam
import scipy as sp
import os

from src.utility.sample import get_patient_id, get_sample_id
from scipy.sparse import csr_matrix

# Caching setup: dictionary to store coverage arrays for regions
coverage_cache: dict[tuple[str, int], list[int]] = ({})  # Key: (contig, segment_start); Value: coverage array for that segment
CACHE_SEGMENT_SIZE = 10_000 # e.g., cache coverage in 1000-base segments to reuse across close positions
MAX_SEGMENTS_CACHED = 20  # Limit cache size to avoid memory blow-up in extremely deep coverage scenarios
current_cache_contig = None  # Track current contig in cache for easy invalidation
bamfile_cache = {}  # Cache for opened BAM files

from dataclasses import dataclass, field
import polars as pl
from pathlib import Path

@dataclass
class ChromStats:
    chrom: str = None
    n_reads: int = 0
    n_mismatches: int = 0
    n_matches: int = 0
    n_indels: int = 0
    n_total: int = 0
    n_skipped: int = 0
    n_invalid_fragments: int = 0

    def __iadd__(self, other: "ChromStats"):
        self.n_reads += other.n_reads
        self.n_mismatches += other.n_mismatches
        self.n_matches += other.n_matches
        self.n_indels += other.n_indels
        self.n_total += other.n_total
        self.n_skipped += other.n_skipped
        self.n_invalid_fragments += other.n_invalid_fragments
        return self

    def __str__(self):
        return (f"[{self.chrom}] reads={self.n_reads:,}, mismatches={self.n_mismatches:,}, "
                f"matches={self.n_matches:,}, indels={self.n_indels:,}, "
                f"opportunities={self.n_total:,}, skipped={self.n_skipped:,}, "
                f"invalid={self.n_invalid_fragments:,}")
            
@dataclass
class GenomeStats:
    """
    Holds aggregated stats for an entire BAM (all chromosomes),
    along with per-chromosome breakdowns.
    """
    per_chrom: Dict[str, ChromStats] = field(default_factory=dict)
    total: ChromStats = field(default_factory=lambda: ChromStats(chrom="ALL"))

    def add(self, stats: ChromStats):
        """Add per-chromosome stats and update total."""
        self.per_chrom[stats.chrom] = stats
        self.total += stats

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
class ReadFeatureRow:
    qname: str
    chr: str
    genomic_pos: int

    obs_type: str
    obs_base: str
    obs_ref: str
    obs_alt_len: int
    obs_ref_len: int
    len_delta: int

    strand: str
    first_in_pair: int
    read_index: int
    fragment_len: int
    seq_length: int

    # Context
    trinucleotide_ctx: str
    start_ctx: str
    end_ctx: str
    reference_ctx: str
    local_gc: float
    local_complexity_1: float
    local_complexity_2: float

    # Error / alignment feature
    n_other_err: int

    # Per-position features
    base_qual_site: int
    dist_to_end: int
    homopolymer_run_len: int

    # Coverage
    read_depth: int

    # Replication
    replication_timing: float

@dataclass
class FilterContext:
    """
    Container for all blacklist and coverage-related filtering resources
    for a single patient.
    """
    genomic_blacklist_by_chr: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None
    donor_blacklist_by_chr: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None
    patient_blacklists_by_chr: Optional[Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]] = None

@dataclass(frozen=True)
class ChromatinResources:
    consensus_masks: Dict[str, Dict[str, csr_matrix]]
    
@dataclass(frozen=True)
class ChromosomeTask:
    chrom: str
    bam_path: str
    fasta_path: str
    mutation_types: list[str]
    bed_intervals: list | None
    config: dict
    chromatin: ChromatinResources | None
    coverage_bw: Any
    replication_bw: Any
    filters: FilterContext | None
    tmp_dir: pathlib.Path

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
    
def calc_string_entropy_k_mer(
    s: str, k: int = 2, alphabet: List[str] = ["A", "C", "G", "T", "N"]
) -> float:
    """
    Calculate Shannon entropy for k-mers in a string.

    This function computes the Shannon entropy for all possible k-mers within a given string.
    """
    k_mers = ["".join(p) for p in product(alphabet, repeat=k)]
    s_length = len(s) - (k - 1)
    if s_length <= 0:
        return 0.0  # No k-mers to count

    # Count occurrences using Counter for efficiency
    counts = dict.fromkeys(k_mers, 0)
    for i in range(s_length):
        k_mer = s[i : i + k]
        if k_mer in counts:
            counts[k_mer] += 1

    total_counts = sum(counts.values())
    frequencies = (
        np.array(list(counts.values())) / total_counts
        if total_counts > 0
        else np.zeros(len(counts))
    )
    H = -np.sum(frequencies * np.log10(frequencies, where=frequencies > 0))
    return float(H)

def calc_string_entropy_k_mer_fast(s: str, k: int = 2) -> float:
    """Fast Shannon entropy for k-mers in DNA strings."""
    n = len(s)
    if n < k:
        return 0.0

    # Count all k-mers
    counts = Counter(s[i:i+k] for i in range(n - k + 1))
    total = sum(counts.values())

    # Shannon entropy in bits (log base 2)
    H = 0.0
    for v in counts.values():
        p = v / total
        H -= p * log2(p)
    return H

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

def count_other_errors_in_read(md_tag: str) -> int:
    """
    Count the number of mismatches in the MD tag.
    """
    mismatch_count = 0
    in_deletion = False

    for c in md_tag:
        if c == "^":
            # Entering a deletion block
            in_deletion = True
        elif c.isdigit():
            # Digits signify match counts, and also end any deletion block
            in_deletion = False
        elif c.isalpha():
            # Alphabetic characters are either mismatches or deleted bases
            if not in_deletion:
                # If we're not in a deletion block, it's a mismatch
                mismatch_count += 1
            # If we are in a deletion block, this char is a deleted base, not a mismatch.

    return mismatch_count

def contains_indel_or_clip(n_ins, n_del, n_soft, n_hard) -> bool:
    return (n_ins + n_del + n_soft + n_hard) > 0

def count_other_errors_in_read_fast(md_tag: str) -> int:
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

def count_cigar_events(cigar):
    """
    Count insertions, deletions, soft clips, hard clips in a CIGAR.
    Returns:
        n_ins, n_del, n_softclip, n_hardclip
    """
    n_ins = n_del = n_soft = n_hard = 0

    if not cigar:
        return 0, 0, 0, 0

    for op, length in cigar:
        if op == 1:       # insertion
            n_ins += 1
        elif op == 2:     # deletion
            n_del += 1
        elif op == 4:     # soft clip
            n_soft += 1
        elif op == 5:     # hard clip
            n_hard += 1

    return n_ins, n_del, n_soft, n_hard

def count_insertions(cigar_string: str) -> int:
    """
    Count the number of inserted bases in a CIGAR string.
    """
    return len(re.findall(r"(\d+)I", cigar_string))

def count_deletions(cigar_string: str) -> int:
    """
    Count the number of deleted bases in a CIGAR string.
    """
    return len(re.findall(r"(\d+)D", cigar_string))

def count_insertions_from_tuples(cigartuples: list[tuple[int, int]]) -> int:
    return sum(length for op, length in cigartuples if op == 1)

def count_deletions_from_tuples(cigartuples: list[tuple[int, int]]) -> int:
    return sum(length for op, length in cigartuples if op == 2)

def parse_leading_trailing_clips(cigar_str: str, clip_op="S|H") -> Tuple[int, int]:
    """
    Given a CIGAR string, return (#clips_at_start, #clips_at_end) for
    either soft/hard clips (default) or whichever ops are provided in clip_op (regex style).
    Example: "10S90M5H" -> leading=10, trailing=5
    """
    if cigar_str is None:
        return (0, 0)
    # Leading
    match_start = re.match(rf"^(\d+)([{clip_op}])", cigar_str)
    start_clip = int(match_start.group(1)) if match_start else 0

    # Trailing
    match_end = re.search(rf"(\d+)([{clip_op}])$", cigar_str)
    end_clip = int(match_end.group(1)) if match_end else 0

    return (start_clip, end_clip)

def trim_umi_for_clips(cd, ce, start_clip, end_clip):
    """
    Given UMI arrays cd, ce, trim from the start and end by the specified amounts.
    If the read is on reverse strand, we assume cd/ce are *already reversed*.
    """
    if cd is None or ce is None:
        return cd, ce
    # If total length < start_clip + end_clip, clamp to empty.
    length_cd = len(cd)
    length_ce = len(ce)
    if length_cd < (start_clip + end_clip) or length_ce < (start_clip + end_clip):
        return [], []

    cd = cd[start_clip : (length_cd - end_clip)]
    ce = ce[start_clip : (length_ce - end_clip)]
    return cd, ce


def trim_sequence_for_clips(sequence, start_clip, end_clip):
    """Trim soft/hard clipped bases from the read sequence at the start/end."""
    if not sequence:
        return sequence
    seq_len = len(sequence)
    if seq_len < (start_clip + end_clip):
        return ""
    return sequence[start_clip : seq_len - end_clip]


def remove_leading_trailing_clips(cigar_string: str, clip_op="SH") -> str:
    """
    Removes all consecutive leading and trailing clip ops from the CIGAR string.
    For example, "10H90M10S5H" -> "90M" when clip_op="SH".
    """
    if cigar_string is None:
        return ""

    # Regex for one or more consecutive <number><op> at the start
    # e.g. 10H or 5S or 10H5S, etc.
    leading_pattern = rf"^(?:\d+[{clip_op}])+"
    # Regex for one or more consecutive <number><op> at the end
    trailing_pattern = rf"(?:\d+[{clip_op}])+$"

    # Remove all consecutive leading clips
    cigar_string = re.sub(leading_pattern, "", cigar_string)
    # Remove all consecutive trailing clips
    cigar_string = re.sub(trailing_pattern, "", cigar_string)

    return cigar_string


def read_filters(
    read: pysam.AlignedSegment,
    mapping_quality_threshold: int = 60,
    min_frag_len: int = 30,
    max_frag_lent: int = 700,
) -> bool:
    """
    Return True only for reads that satisfy the project's inclusion rules.

    * mapped, primary (not secondary/supplementary)
    * paired and properly paired
    * mapping quality ≥ mapping_quality_threshold
    * fragment size 30 - 700 bp
    """
    if (
        read.is_unmapped
        or read.is_secondary
        or read.is_supplementary
        or not read.is_paired
        or not read.is_proper_pair
        or read.mapping_quality < mapping_quality_threshold
        or abs(read.template_length) < min_frag_len
        or abs(read.template_length) > max_frag_lent
    ):
        return False
    return True

def get_bam_handle(bam_path: str) -> pysam.AlignmentFile:
    """Return a cached pysam.AlignmentFile handle or open one if not already cached."""
    if bam_path not in bamfile_cache:
        if not os.path.exists(bam_path):
            raise FileNotFoundError(f"BAM file not found: {bam_path}")
        # Use a small number of threads; pysam threads control decompression, not compute
        bamfile_cache[bam_path] = pysam.AlignmentFile(bam_path, "rb", threads=2)
    return bamfile_cache[bam_path]

def get_read_depth(bam_path: str, contig: str, position: int) -> int:
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
    segment_key = (contig, segment_start)

    # Try to reuse cached coverage for this segment
    segment_depth = coverage_cache.get(segment_key)
    if segment_depth is None:
        segment_end = segment_start + CACHE_SEGMENT_SIZE

        # Compute depth arrays for this chunk
        A, C, G, T = bam_file.count_coverage(
            contig,
            start=segment_start,
            stop=segment_end,
            read_callback=read_filters  # <- user-supplied filter function
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
    
def get_read_depth_from_read(
    read: pysam.AlignedSegment,
    position: int,
    bam_file: Optional[pysam.AlignmentFile] = None,
) -> int:
    """
    Return read depth at a 0-based genomic position using an existing read
    or BAM handle, avoiding repeated file reopen calls.

    Parameters
    ----------
    read : pysam.AlignedSegment
        The read to use for extracting contig information.
    position : int
        0-based genomic coordinate.
    bam_file : Optional[pysam.AlignmentFile]
        Optional open BAM handle. If not given, will use a global cache
        from get_bam_handle(read).bam_path if available.

    Returns
    -------
    int
        Read depth at the given position.
    """
    global current_cache_contig, coverage_cache

    contig = read.reference_name
    if contig is None:
        return 0

    # Use provided handle or retrieve from cache
    if bam_file is None:
        try:
            bam_file = get_bam_handle(read.header.to_dict()["HD"]["SO"])
        except Exception:
            return 0

    # Reset coverage cache when contig changes
    if contig != current_cache_contig:
        coverage_cache.clear()
        current_cache_contig = contig

    # Chunk-based cache indexing
    segment_start = (position // CACHE_SEGMENT_SIZE) * CACHE_SEGMENT_SIZE
    segment_key = (contig, segment_start)

    segment_depth = coverage_cache.get(segment_key)
    if segment_depth is None:
        segment_end = segment_start + CACHE_SEGMENT_SIZE
        A, C, G, T = bam_file.count_coverage(
            contig,
            start=segment_start,
            stop=segment_end,
            read_callback=read_filters,
        )
        segment_depth = [a + c + g + t for a, c, g, t in zip(A, C, G, T)]
        coverage_cache[segment_key] = segment_depth

        # Evict oldest cache if necessary
        if len(coverage_cache) > MAX_SEGMENTS_CACHED:
            oldest_key = next(iter(coverage_cache))
            coverage_cache.pop(oldest_key, None)

    idx = position - segment_start
    return segment_depth[idx] if 0 <= idx < len(segment_depth) else 0

def rows_to_polars(rows, schema=None):
    n = len(rows)
    first = rows[0].__dict__.keys()

    cols = {k: [None]*n for k in first}

    for i, row in enumerate(rows):
        d = row.__dict__
        for k, v in d.items():
            cols[k][i] = v

    return pl.DataFrame(cols, schema=schema)

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