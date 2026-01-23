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
from typing import Dict, Iterable, List, Optional, Tuple
import multiprocessing as mp
from bisect import bisect_right

import numpy as np
import polars as pl
import pysam
import gc
import psutil

from collections import defaultdict

from src.features.constants import _MD_RE
from src.utility.data import FilterContext

def perfect_match_in_overlap(read: pysam.AlignedSegment,
                             overlap: tuple[int, int],
                             highqual_positions: Optional[set[int]] = None) -> bool:
    """
    Return True if the read perfectly matches the reference across the overlap:
    no mismatches and no indels intersecting the region.
    """
    start, end = overlap
    if start >= end:
        return False

    # Any mismatch disqualifies
    if mismatch_in_overlap(read, overlap, highqual_positions):
        return False

    # Any indel overlapping disqualifies
    if indel_in_overlap(read, overlap, highqual_positions):
        return False

    return True

def _extract_read_end_context(read, k: int = 5):
    """
    Extract the first k bases (start_context) and the last k bases
    (end_context) from a single read's query sequence.

    Parameters
    ----------
    read : pysam.AlignedSegment
        A single sequencing read.
    k : int
        Number of bases to extract from each end of the read.

    Returns
    -------
    start_context : str
        First k bases of the read.
    end_context : str
        Last k bases of the read.
    """

    seq = read.query_sequence
    if seq is None:
        return None, None

    seq = seq.upper()

    start_context = seq[:k]
    end_context   = seq[-k:]

    return start_context, end_context

def _extract_fragment_end_context(read, mate, k: int = 5):
    """
    Extract the first k bases of the fragment (start_context)
    and the last k bases of the fragment (end_context),
    using genomic coordinates to correctly determine fragment
    orientation and boundaries.

    Parameters
    ----------
    read, mate : pysam.AlignedSegment
        The two reads of a paired-end fragment.
    k : int
        Number of bases to extract from each end of the fragment.

    Returns
    -------
    start_context : str
        First k bases of the fragment.
    end_context : str
        Last k bases of the fragment.
    """

    # Identify leftmost and rightmost read based on genome coordinates
    if read.reference_start <= mate.reference_start:
        left = read
        right = mate
    else:
        left = mate
        right = read

    # Extract sequences in uppercase
    left_seq = left.query_sequence.upper()
    right_seq = right.query_sequence.upper()

    # Handle cases where k exceeds read lengths
    start_context = left_seq[:k]
    end_context = right_seq[-k:]

    return start_context, end_context

def mismatch_in_overlap(read: pysam.AlignedSegment,
                        overlap: tuple[int, int],
                        highqual_positions: Optional[set[int]] = None) -> bool:
    """
    Return True if the read contains at least one mismatch within the overlap region.
    Uses the MD tag to quickly infer mismatch positions without parsing CIGAR pairs.

    Parameters
    ----------
    read : pysam.AlignedSegment
        The read to inspect.
    overlap : tuple[int, int]
        (start, end) reference coordinates of the overlapping region, 0-based.

    Returns
    -------
    bool
        True if a mismatch exists in the overlap, False otherwise.
    """
    start, end = overlap
    if start >= end:
        return False
    
    # Get MD tag safely
    try:
        md_tag = read.get_tag("MD")
    except KeyError:
        return False  # no tag, assume no mismatches

    if not md_tag:
        return False
    
    # Decode MD tag structure efficiently
    # Example: "35A10^AC5T0" → mismatches at positions after matched runs
    ref_pos = read.reference_start
    md_iter = _MD_RE.finditer(md_tag)

    for match in md_iter:
        if match.group(1):  # matched bases
            ref_pos += int(match.group(1))
        elif match.group(2):  # single-base mismatch
            if start <= ref_pos < end:
                if highqual_positions is not None:
                    if ref_pos in highqual_positions:
                        return True
                else:
                    return True
            ref_pos += 1
        elif match.group(3):  # deletion in ref
            ref_pos += len(match.group(3))

    return False  # no mismatches in overlap

def indel_in_overlap(read: pysam.AlignedSegment,
                     overlap: tuple[int, int],
                     highqual_positions: Optional[set[int]] = None) -> bool:
    """
    Return True if the read contains an insertion or deletion overlapping
    the given reference interval and (if provided) touching a high-quality
    reference position.
    """
    start, end = overlap
    if start >= end:
        return False

    if read.cigartuples is None:
        return False

    ref_pos = read.reference_start

    for op, length in read.cigartuples:
        if op == 0:  # M
            ref_pos += length

        elif op == 2:  # D (deletion consumes reference)
            del_start = ref_pos
            del_end = ref_pos + length

            # overlap with region?
            if del_start < end and del_end > start:
                if highqual_positions is None:
                    return True
                # check if ANY deleted ref position is high-quality
                for p in range(max(del_start, start), min(del_end, end)):
                    if p in highqual_positions:
                        return True

            ref_pos += length

        elif op == 1:  # I (insertion anchored at ref_pos)
            if start <= ref_pos < end:
                if highqual_positions is None or ref_pos in highqual_positions:
                    return True
            # ref_pos unchanged

        elif op == 3:  # N
            ref_pos += length

        else:
            # S, H, P
            pass

    return False

def _is_identical_to_ref(
    read: pysam.AlignedSegment,
    mate: pysam.AlignedSegment,
    ref_seq: str,
    overlap: tuple[int, int],
) -> bool:
    start, end = overlap
    if start >= end:
        return True

    def is_ref_identical(r: pysam.AlignedSegment) -> bool:
        seq = r.query_sequence
        if not seq:
            return False
        seq_u = seq.upper()
        for qpos, rpos, refb in r.get_aligned_pairs(with_seq=True, matches_only=True):
            if rpos is None or refb is None:
                continue
            if not (start <= rpos < end):
                continue
            if seq_u[qpos] != refb.upper():
                return False
        return True

    return is_ref_identical(read) and is_ref_identical(mate)

def _mates_agree(read: pysam.AlignedSegment,
                 mate: pysam.AlignedSegment,
                 overlap: tuple[int, int]) -> bool:
    """
    Return True if the two mates agree (same base calls) within their overlapping reference region.
    """
    if not overlap:
        return True  # no overlap → nothing to disagree on

    start_ovl, end_ovl = overlap
    seq1, seq2 = read.query_sequence, mate.query_sequence
    if not seq1 or not seq2:
        return False

    # Build dict of ref_pos -> base for each read
    # (get_aligned_pairs returns query and ref coordinates)
    base_map_1 = {rpos: seq1[qpos].upper() for qpos, rpos in read.get_aligned_pairs(matches_only=True)
                  if rpos is not None and start_ovl <= rpos < end_ovl}
    base_map_2 = {rpos: seq2[qpos].upper() for qpos, rpos in mate.get_aligned_pairs(matches_only=True)
                  if rpos is not None and start_ovl <= rpos < end_ovl}

    # Intersection of covered reference positions
    shared_positions = base_map_1.keys() & base_map_2.keys()
    if not shared_positions:
        return False  # both map, but don't actually overlap at aligned bases

    # Check all overlapping bases
    for pos in shared_positions:
        if base_map_1[pos] != base_map_2[pos]:
            return False
        
    return True

def _fragment_is_proper_length(read1: pysam.AlignedSegment,
                               read2: pysam.AlignedSegment,
                               min_len: int,
                               max_len: int) -> bool:
    """
    Return True if the fragment length defined by:
    fragment_length = reverse_read.reference_end - forward_read.reference_start
    lies within [min_len, max_len].
    """    
    if read1.reference_id != read2.reference_id:
        return False
    
    if read1.is_reverse == read2.is_reverse:
        return False
    
    if read1.is_reverse:
        fwd, rev = read2, read1
    else:
        fwd, rev = read1, read2
    
    frag_len = rev.reference_end - fwd.reference_start
    
    if frag_len <= 0:
        return False
    
    return min_len < frag_len <= max_len

def _paired_overlap(read1: pysam.AlignedSegment,
                    read2: pysam.AlignedSegment) -> tuple[int, int] | None:
    """
    Compute 0-based overlap [start, end) on reference between two properly paired reads.
    Return None if no overlap or on different contigs.
    """
    if read1.reference_id != read2.reference_id:
        return None
    start = max(read1.reference_start, read2.reference_start)
    end = min(read1.reference_end, read2.reference_end)
    return (start, end) if end > start else None
