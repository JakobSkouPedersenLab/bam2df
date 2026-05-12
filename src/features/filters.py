import contextlib
import csv
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Callable, Dict, Iterable, List, Optional, Tuple
import multiprocessing as mp
from bisect import bisect_right

import numpy as np
import polars as pl
import pysam
import gc
import psutil

from collections import defaultdict

def read_matches_any_type(read, mutation_types, read_predicates) -> bool:
    for m in mutation_types:
        predicate = read_predicates[m]
        if predicate(read):
            return True
    return False

def read_matches_any_type_in_overlap(read, overlap, highqual_overlap, mutation_types, overlap_predicates):
    for m in mutation_types:
        if overlap_predicates[m](read, overlap, highqual_overlap):
            return True
    return False

def _count_indel_bases(read: pysam.AlignedSegment) -> int:
    """Total inserted+deleted bases from CIGAR (I+D)."""
    if read.cigartuples is None:
        return 0
    return sum(length for op, length in read.cigartuples if op in (1, 2))

def read_has_indel(read: pysam.AlignedSegment) -> bool:
    if read.cigartuples is None:
        return False
    return any(op in (1, 2) for op, _ in read.cigartuples)

def read_has_mismatch(read: pysam.AlignedSegment) -> bool:
    """
    True if the read contains >= 1 substitution mismatch.
    Uses NM (edit distance) minus indel bases as a coarse proxy.

    Note: NM can include some complexities depending on aligner/settings,
    but this matches your current "coarse check" intent.
    """
    try:
        nm = read.get_tag("NM")
    except KeyError:
        return False

    indel_bases = _count_indel_bases(read)
    mismatch_proxy = nm - indel_bases
    return mismatch_proxy > 0

def read_has_match(read: pysam.AlignedSegment) -> bool:
    """
    True if the read contains >= 1 aligned match position.

    Logic: an aligned position is either a match or mismatch.
    aligned_bases = query_alignment_length - inserted_bases
    mismatches ≈ NM - indel_bases
    matches ≈ aligned_bases - mismatches
    """
    # No alignment => no match
    if read.cigartuples is None:
        return False

    # Insertions consume query but don't align to reference positions.
    inserted_bases = sum(length for op, length in read.cigartuples if op == 1)  # I
    aligned_bases = max(0, read.query_alignment_length - inserted_bases)
    if aligned_bases == 0:
        return False

    try:
        nm = read.get_tag("NM")
    except KeyError:
        # If no NM tag, fall back to: any aligned base implies at least one (match or mismatch).
        # We cannot distinguish; but for "has match" it's safer to keep reads rather than drop.
        return True

    indel_bases = _count_indel_bases(read)
    mismatches = max(0, nm - indel_bases)
    matches = aligned_bases - mismatches
    return matches > 0

def _high_quality_overlap(read, mate, overlap, min_bq):
    """
    Compute high-quality overlapping positions between two reads.
    """
    if min_bq is None:
        start, end = overlap
        return list(range(start, end))

    # Build ref_pos -> qpos maps for both reads
    # get_aligned_pairs returns (qpos, rpos)
    map_read = {}
    for qpos, rpos in read.get_aligned_pairs(matches_only=False):
        if rpos is not None and qpos is not None:
            map_read[rpos] = qpos

    map_mate = {}
    for qpos, rpos in mate.get_aligned_pairs(matches_only=False):
        if rpos is not None and qpos is not None:
            map_mate[rpos] = qpos

    start, end = overlap
    highqual = []

    for ref_pos in range(start, end):
        qpos1 = map_read.get(ref_pos)
        qpos2 = map_mate.get(ref_pos)

        if qpos1 is None or qpos2 is None:
            continue

        bq1 = read.query_qualities[qpos1]
        bq2 = mate.query_qualities[qpos2]

        if bq1 >= min_bq and bq2 >= min_bq:
            highqual.append(ref_pos)

    return set(highqual)

def _high_quality_single_read(read, min_bq):
    """
    Identify high-quality reference positions for a single read.
    Equivalent to paired version but without requiring a mate.
    """
    # If no threshold, include all aligned reference positions
    if min_bq is None:
        return [
            rpos 
            for qpos, rpos in read.get_aligned_pairs(matches_only=False)
            if rpos is not None and qpos is not None
        ]

    highqual = []
    for qpos, rpos in read.get_aligned_pairs(matches_only=False):
        if rpos is None or qpos is None:
            continue
        if read.query_qualities[qpos] >= min_bq:
            highqual.append(rpos)

    return highqual

def _read_filters(read: pysam.AlignedSegment,
                  min_mapq: int,
                  min_frag_len: int = None,
                  max_frag_len: int = None) -> bool:
    """
    Return True if a read passes all inclusion criteria.
    """
    # Basic read-level flags
    if (
        read.is_unmapped
        or read.is_secondary
        or read.is_supplementary
        or read.is_duplicate
        or read.is_qcfail
        or not read.is_paired
    ):
        return False
    
    # Skip reads with missing qualities
    quals = read.query_qualities
    if quals is None or len(quals) == 0:
        return False

    # Mapping quality
    if read.mapping_quality < min_mapq:
        return False
    
    # Fragment length filter
    if min_frag_len is not None or max_frag_len is not None:
        frag_len = abs(read.template_length)

        if frag_len == 0:
            return False

        if min_frag_len is not None and frag_len < min_frag_len:
            return False
        if max_frag_len is not None and frag_len > max_frag_len:
            return False
        
    return True

def _load_bed(bed_path: str) -> Dict[str, List[Tuple[int, int]]]:
    """
    BED parser
    """
    bed_path = Path(bed_path)

    # Read as Polars DataFrame
    df = pl.read_csv(
        bed_path,
        separator="\t",
        has_header=False,
        new_columns=["chrom", "start", "end"],
        dtypes={"chrom": pl.Utf8, "start": pl.Int64, "end": pl.Int64},
    )

    # Group by chromosome and collect start/end tuples
    bed_dict: Dict[str, List[Tuple[int, int]]] = {
        chrom: list(zip(group["start"].to_list(), group["end"].to_list()))
        for chrom, group in df.group_by("chrom")
    }

    return bed_dict

def _make_interval_lookup(bed_dict: Dict[str, List[Tuple[int, int]]]):
    """
    Return a dict: chrom → (start_array, end_array) **both 1-based, inclusive**.
    """
    lookup: Dict[str, Tuple[List[int], List[int]]] = {}
    for chrom, intervals in bed_dict.items():
        if not intervals:
            continue
        pairs = sorted((s + 1, e) for s, e in intervals)  # convert to 1-based & sort
        starts = [p[0] for p in pairs]
        ends   = [p[1] for p in pairs]
        lookup[chrom] = (starts, ends)
    return lookup

def _in_bed_1based(chrom: str, pos1: int, lookup) -> bool:
    """True iff **pos1** (1-based) lies inside any interval on *chrom*."""
    if chrom not in lookup:
        return False
    starts, ends = lookup[chrom]
    i = bisect_right(starts, pos1) - 1
    return i >= 0 and pos1 <= ends[i]
