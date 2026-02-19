import math
import pysam
from typing import Any, List, Iterable, Optional, Tuple, Dict
import bisect
import numpy as np

import pyBigWig
import hashlib

from src.features.constants import _MD_RE, _MD_TOKEN_RE
from src.utility.data import FilterContext, ReferenceCache, calculate_local_complexity, get_read_depth

def match_in_overlap(read: pysam.AlignedSegment,
                     overlap: tuple[int, int],
                     highqual_positions: Optional[set[int]] = None) -> bool:
    """
    True if the read contains at least one *matching* reference position within the overlap.
    Uses the MD tag to detect match-runs.
    """
    start, end = overlap
    if start >= end:
        return False

    try:
        md_tag = read.get_tag("MD")
    except KeyError:
        return False  # can't safely assert matches without MD in this scheme

    if not md_tag:
        return False

    ref_pos = read.reference_start

    for m in _MD_RE.finditer(md_tag):
        if m.group(1):  # run of matches
            run_len = int(m.group(1))
            run_start = ref_pos
            run_end = ref_pos + run_len

            # intersect with overlap?
            a = max(run_start, start)
            b = min(run_end, end)
            if a < b:
                if highqual_positions is None:
                    return True
                # need at least one HQ position inside intersection
                for p in range(a, b):
                    if p in highqual_positions:
                        return True

            ref_pos = run_end

        elif m.group(2):  # single-base mismatch
            ref_pos += 1

        elif m.group(3):  # deletion (consumes reference)
            ref_pos += len(m.group(3))

    return False

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

def _calculate_fragment_length(read1: pysam.AlignedSegment,
                               read2: pysam.AlignedSegment) -> bool:
    """
    Calculate fragment length
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
    
    return frag_len

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
    
    return min_len <= frag_len <= max_len

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

def qname_u64(qname: str) -> int:
    # 64-bit stable hash (unsigned)
    return int.from_bytes(
        hashlib.blake2b(qname.encode("utf-8"), digest_size=8).digest(),
        byteorder="little",
        signed=False,
    )

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
            
def _iter_mismatches(read: pysam.AlignedSegment):
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
    replication_bw: Optional["pyBigWig.bigWigFile"] = None
) -> Optional[Dict[str, Any]]:
    """Per-base feature extraction."""
    # BED filter
    if bed_lookup and not _in_bed_1based(chrom, pos_1based, bed_lookup):
        return None

    ctx_len = 2 * flank + 1
    ref_ctx = ref_cache.fetch(chrom, pos_1based, flank)
    if len(ref_ctx) < ctx_len:
        return None

    ref_center = ref_ctx[flank]
    ctx_m1 = ref_ctx[flank - 1]
    ctx_p1 = ref_ctx[flank + 1]
    lc1, lc2 = calculate_local_complexity(ref_ctx)
    gc_frac = (ref_ctx.count("G") + ref_ctx.count("C")) / ctx_len
    n_other_err = other_err - (1 if obs_type == "S" else 0)

    # Coverage
    read_depth = get_read_depth(bam_path, chrom, pos_1based - 1)
        
    # Replication timing
    replication_timing = None
    if replication_bw is not None:
        rt = replication_bw.values(chrom, pos_1based - 1, pos_1based)
        if rt and rt[0] is not None and not math.isnan(rt[0]):
            replication_timing = float(rt[0])
        else:
            replication_timing = None
    
    base_qual_site = _get_base_qual_at(read, query_pos)
    dist_to_end = _get_dist_to_read_end(read, query_pos)
    dist_to_indel_or_clip = _get_dist_to_indel_or_clip(read, pos_1based - 1)
    homopolymer_run_len = _get_homopolymer_run_length(ref_cache, chrom, pos_1based)
    
    # Allele-length features
    obs_alt_len = len(obs_base) if obs_base is not None else 0
    obs_ref_len = len(obs_ref) if obs_ref is not None else 0
    len_delta = obs_alt_len - obs_ref_len
            
    row = {
        "qname": qname_u64(read.query_name),
        "chr": chrom,
        "genomic_pos": pos_1based,

        "obs_type": obs_type,
        "obs_base": obs_base,
        "obs_ref": obs_ref,
        "obs_alt_len": obs_alt_len,
        "obs_ref_len": obs_ref_len,
        "len_delta": len_delta,

        "strand": strand,
        "first_in_pair": int(read.is_read1),
        "read_index": query_pos - read.query_alignment_start + 1,
        "fragment_len": fragment_len,
        "seq_length": read.query_alignment_length,

        "trinucleotide_ctx": f"{ctx_m1}{ref_center}{ctx_p1}",
        "start_ctx": start_ctx,
        "end_ctx": end_ctx,
        "reference_ctx": ref_ctx,
        "local_gc": gc_frac,
        "local_complexity_1": lc1,
        "local_complexity_2": lc2,

        "n_other_err": n_other_err,

        "base_qual_site": base_qual_site,
        "dist_to_end": dist_to_end,
        "dist_to_indel_or_clip": dist_to_indel_or_clip,
        "homopolymer_run_len": homopolymer_run_len,

        "read_depth": read_depth,
    }
    
    if replication_timing is not None:
        row["replication_timing"] = replication_timing
        
    return row
    