import math
import random
import numpy as np
import pysam
from typing import List, Iterable, Optional, Tuple, Dict
import re
import bisect
import numpy as np
import polars as pl

from collections import Counter
from math import log2

from src.utility.data import (FilterContext, ReadFeatureRow, ReferenceCache, calculate_local_complexity, count_deletions, count_deletions_from_tuples,
                              count_insertions, count_insertions_from_tuples, count_other_errors_in_read, 
                              count_other_errors_in_read_fast, get_read_depth, get_read_depth_from_read)

import pyBigWig

_MD_TOKEN_RE = re.compile(r"(\d+)|\^([A-Z]+)|([A-Z])")

def _in_bed_1based(chrom: str, pos1: int, lookup) -> bool:
    """True iff **pos1** (1-based) lies inside any interval on *chrom*."""
    if chrom not in lookup:
        return False
    starts, ends = lookup[chrom]
    i = bisect.bisect_right(starts, pos1) - 1
    return i >= 0 and pos1 <= ends[i]

def _iter_mismatches(read: pysam.AlignedSegment):
    """Yield (qpos, rpos, ref_base) for true mismatches from MD tag."""
    seq = read.query_sequence
    if seq is None or not read.has_tag("MD"):
        return

    md = read.get_tag("MD")
    ref_pos = read.reference_start
    query_pos = 0

    for m in _MD_TOKEN_RE.finditer(md):
        if m.group(1):           # match run
            n = int(m.group(1))
            ref_pos += n
            query_pos += n
        elif m.group(2):         # deletion in reference
            ref_pos += len(m.group(2))
        elif m.group(3):         # mismatch
            ref_base = m.group(3)
            if query_pos < len(seq):     # avoid overrun
                yield query_pos, ref_pos, ref_base
            ref_pos += 1
            query_pos += 1
            
def _iter_mismatches_non_re(read: pysam.AlignedSegment):
    seq = read.query_sequence
    if not seq:
        return
    for qpos, rpos, rbase in read.get_aligned_pairs(with_seq=True, matches_only=True):
        if rpos is None or rbase is None or qpos is None:
            continue
        if seq[qpos].upper() != rbase.upper():
            yield qpos, rpos, rbase
            
def _pos_in_blacklist_chr(arr_by_chr: dict[str, tuple[np.ndarray, np.ndarray]],
                          chrom: str, pos: int) -> bool:
    """
    O(log M) lookup using binary search on pre-sorted arrays.
    """
    if chrom not in arr_by_chr:
        return False
    starts, ends = arr_by_chr[chrom]
    i = bisect.bisect_right(starts, pos) - 1
    return bool(i >= 0 and pos < ends[i])

def _is_filtered(filters: "FilterContext", chrom: str, pos: int, patient_id: str) -> bool:
    if not filters:
        return False
    
    genomic_blacklist_by_chr = filters.genomic_blacklist_by_chr
    donor_blacklist_by_chr = filters.donor_blacklist_by_chr
    patient_blacklists_by_chr = filters.patient_blacklists_by_chr

    if genomic_blacklist_by_chr and _pos_in_blacklist_chr(genomic_blacklist_by_chr, chrom, pos):
        return True

    if donor_blacklist_by_chr and _pos_in_blacklist_chr(donor_blacklist_by_chr, chrom, pos):
        return True
    
    if patient_blacklists_by_chr:
        for key in ("snp_blacklist", "lowpower_blacklist", "problematic_blacklist"):
            chr_dict = patient_blacklists_by_chr.get(key)
            if chr_dict and _pos_in_blacklist_chr(chr_dict, chrom, pos):
                return True
    
    return False

def _get_base_qual_at(read: pysam.AlignedSegment, qpos: int) -> Optional[int]:
    """Return base quality at query position or None."""
    try:
        return read.query_qualities[qpos]
    except Exception:
        return None
    
def _get_dist_to_read_end(read: pysam.AlignedSegment, qpos: int) -> int:
    """
    Distance (in bases) from the mismatch site to the nearest aligned end.
    Robust to soft-clipped prefixes because pysam gives alignment coords.
    """
    left = qpos - read.query_alignment_start
    right = read.query_alignment_end - 1 - qpos
    return min(left, right)

def _get_dist_to_indel_or_clip(
    read: pysam.AlignedSegment,
    ref_pos0: int
) -> Optional[int]:
    """
    Return minimum distance (in reference coordinates) from the mismatch
    to any insertion (I), deletion (D), or soft-clip (S) position.
    Returns None if no such event is present.
    """
    best = None
    rpos = read.reference_start
    qpos = read.query_alignment_start

    for op, length in (read.cigartuples or []):
        if op in (0, 7, 8):  # M, =, X (consumes ref + query)
            # If mismatch inside this block, distance to ends is a proxy
            if rpos <= ref_pos0 < rpos + length:
                d = min(ref_pos0 - rpos, (rpos + length - 1) - ref_pos0)
                best = d if best is None else min(best, d)
            rpos += length
            qpos += length

        elif op in (1, 2, 4):  # I, D, S all signal a local alignment event
            d = abs(ref_pos0 - rpos)
            best = d if best is None else min(best, d)
            if op == 1:
                qpos += length
            elif op == 2:
                rpos += length
            else:
                pass  # soft clip does not consume reference

        else:
            # N / padding, etc.
            if op == 3:  # N: consumes ref
                rpos += length
            else:
                qpos += length

    return best

def _get_homopolymer_run_length(
    ref_cache: ReferenceCache, chrom: str, pos_1based: int
) -> int:
    """
    Compute run length of identical bases crossing the central position.
    Uses ±50 bp window for safety.
    """
    window = ref_cache.fetch(chrom, pos_1based, flank=50)  # returns 101 bp
    center = 50
    base = window[center]

    left = 0
    i = center - 1
    while i >= 0 and window[i] == base:
        left += 1
        i -= 1

    right = 0
    i = center + 1
    while i < len(window) and window[i] == base:
        right += 1
        i += 1

    return left + 1 + right
          
def _build_read_row(
    read: pysam.AlignedSegment,
    bam_path: str,
    chrom: str,
    pos_1based: int,
    obs_type: str,   # "S" / "I" / "D", "M"
    obs_base: str,   # alt allele string (>=1)
    obs_ref: str,    # ref allele string (>=1)
    strand: str,
    fragment_len: int,
    query_pos: int,
    ref_cache: ReferenceCache,
    flank: int,
    start_ctx: str,
    end_ctx: str,
    other_err: int = 0,
    bed_lookup: Optional[Dict[str, Tuple[List[int], List[int]]]] = None,
    coverage_bw: Optional["pyBigWig.bigWigFile"] = None,
    replication_bw: Optional["pyBigWig.bigWigFile"] = None
) -> Optional[ReadFeatureRow]:
    """Optimized per-base feature extraction (CPU + I/O efficient)."""
    # BED filter
    if bed_lookup and not _in_bed_1based(chrom, pos_1based, bed_lookup):
        return None

    ctx_len = 2 * flank + 1
    ref_ctx = ref_cache.fetch(chrom, pos_1based, flank)
    if len(ref_ctx) < ctx_len:
        return None

    ctx_m1, ctx_p1 = ref_ctx[flank - 1], ref_ctx[flank + 1]
    lc1, lc2 = calculate_local_complexity(ref_ctx)
    gc_frac = (ref_ctx.count("G") + ref_ctx.count("C")) / ctx_len

    md_tag_raw = read.get_tag("MD")
    md_tag = str(md_tag_raw) if md_tag_raw is not None else ""
    # subtract 1 only for SNVs
    other_err = count_other_errors_in_read(md_tag) - (1 if obs_type == "S" else 0)
    
    # Coverage
    if coverage_bw is not None:
        cov = coverage_bw.values(chrom, pos_1based - 1, pos_1based)
        if cov and cov[0] is not None and not math.isnan(cov[0]):
            read_depth = int(round(cov[0]))
        else:
            read_depth = 0
    else:
        read_depth = get_read_depth(bam_path, chrom, pos_1based - 1)
        
    # Replication timing
    if replication_bw is not None:
        rt = replication_bw.values(chrom, pos_1based - 1, pos_1based)
        if rt and rt[0] is not None and not math.isnan(rt[0]):
            replication_timing = float(rt[0])
        else:
            replication_timing = np.nan
    else:
        replication_timing = np.nan
    
    base_qual_site = _get_base_qual_at(read, query_pos)
    dist_to_end = _get_dist_to_read_end(read, query_pos)
    dist_to_indel_or_clip = _get_dist_to_indel_or_clip(read, pos_1based - 1)
    homopolymer_run_len = _get_homopolymer_run_length(ref_cache, chrom, pos_1based)
    
    # Allele-length features
    obs_alt_len = len(obs_base) if obs_base is not None else 0
    obs_ref_len = len(obs_ref) if obs_ref is not None else 0
    len_delta = obs_alt_len - obs_ref_len
            
    return ReadFeatureRow(
        qname=read.query_name,
        chr=chrom,
        genomic_pos=pos_1based,

        # Allele
        obs_type=obs_type,
        obs_base=obs_base,
        obs_ref=obs_ref,
        obs_alt_len=obs_alt_len,
        obs_ref_len=obs_ref_len,
        len_delta=len_delta,

        strand=strand,
        first_in_pair=int(read.is_read1),
        read_index=query_pos - read.query_alignment_start + 1,
        fragment_len=fragment_len,
        seq_length=read.query_alignment_length,

        # Context
        trinucleotide_ctx=f"{ctx_m1}{ref_ctx[flank]}{ctx_p1}",
        start_ctx=start_ctx,
        end_ctx=end_ctx,
        reference_ctx=ref_ctx,
        local_gc=gc_frac,
        local_complexity_1=lc1,
        local_complexity_2=lc2,

        # Error / alignment features
        n_other_err=other_err,

        # Per-position features
        base_qual_site=base_qual_site,
        dist_to_end=dist_to_end,
        homopolymer_run_len=homopolymer_run_len,

        # Coverage
        read_depth=read_depth,

        # Replication
        replication_timing=replication_timing
    )