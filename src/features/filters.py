from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from bisect import bisect_right

import polars as pl
import pysam

def _read_has_mismatches(read: pysam.AlignedSegment) -> bool:
    """
    Return True if the read contains one or more mismatches, insertions, or deletions
    according to the NM tag. Returns False if NM == 0 or tag is missing.

    NM (edit distance) counts mismatches + indels vs. reference.
    """
    try:
        nm = read.get_tag("NM")
        return nm > 0
    except KeyError:
        # Missing NM tag — conservative choice: assume no mismatches
        return False

def _high_quality_overlap(read, mate, overlap, min_bq):
    """
    Identify high-quality overlapping reference positions between two reads.
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

    return highqual

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
                  extract_type: str,
                  min_mapq: int,
                  max_nm: int) -> bool:
    """
    Return True if a read passes all inclusion criteria.
    """
    cigar_stats = read.get_cigar_stats()[0]
    
    # Check for clips
    has_clip  = sum(cigar_stats[4:6]) != 0 # S, H (BAM_CSOFT_CLIP, BAM_CHARD_CLIP)
    if has_clip:
        return False
    
    # Check for indels
    if extract_type != "indel":
        has_indel = sum(cigar_stats[1:4]) != 0 # I, D, N (BAM_CINS, BAM_CDEL, BAM_CREF_SKIP)
        if has_indel:
            return False
        
    # Basic read-level flags
    if (
        read.is_unmapped
        or read.is_secondary
        or read.is_supplementary
        or read.is_duplicate
        or read.is_qcfail
        or not read.is_paired
        or not read.is_proper_pair
    ):
        return False
    
    # Skip reads with missing qualities
    quals = read.query_qualities
    if quals is None or len(quals) == 0:
        return False

    # Mapping quality
    if read.mapping_quality < min_mapq:
        return False

    # Edit distance (NM tag)
    if max_nm is not None:
        try:
            nm = read.get_tag("NM")
            if nm > max_nm:
                return False
        except KeyError:
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
