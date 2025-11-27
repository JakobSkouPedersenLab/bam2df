import random
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union
import pysam
import pyBigWig

from src.features.filters import _in_bed_1based
from src.features.reference import ReferenceCache
from src.utility.data import (FilterContext, ReadFeatureRow, count_cigar_events, count_other_errors_in_read_fast)
from src.features.read import _build_read_row, _is_filtered, _iter_mismatches_non_re

def extract_read_features(
    read: pysam.AlignedSegment,
    bam_path: str,
    patient_id: str,
    ref_cache: ReferenceCache,
    extract_type: str,
    context_flank: int,
    start_context: str,
    end_context: str,
    overlap: Optional[Tuple[int, int]] = None,
    bed_lookup: Optional[Dict[str, Tuple[List[int], List[int]]]] = None,
    coverage_bw: Optional["pyBigWig.bigWigFile"] = None,
    filters: Optional[FilterContext] = None,
    highqual_positions: Optional[set[int]] = None,
) -> Tuple[Iterator[ReadFeatureRow], int]:
    """Yield one or more feature rows for the given read."""
    seq = read.query_sequence
    seq_upper = seq.upper()
    chrom_str = read.reference_name or ""
    strand = "fwd" if not read.is_reverse else "rev"
    frag_len = abs(read.template_length)
    
    # Route by extract type
    if extract_type == "indel":
        return _extract_indels_from_read(
            read=read,
            bam_path=bam_path,
            ref_cache=ref_cache,
            chrom_str=chrom_str,
            strand=strand,
            frag_len=frag_len,
            context_flank=context_flank,
            start_context=start_context,
            end_context=end_context,
            bed_lookup=bed_lookup,
            coverage_bw=coverage_bw
        )

    if extract_type == "match":
        return _extract_matches_from_read(
            read=read,
            bam_path=bam_path,
            ref_cache=ref_cache,
            chrom_str=chrom_str,
            strand=strand,
            frag_len=frag_len,
            context_flank=context_flank,
            start_context=start_context,
            end_context=end_context,
            bed_lookup=bed_lookup,
            seq_upper=seq_upper,
            coverage_bw=coverage_bw,
            highqual_positions=highqual_positions
        )

    if extract_type == "mismatch":
        return _extract_mismatches_from_read(
            read=read,
            bam_path=bam_path,
            patient_id=patient_id,
            ref_cache=ref_cache,
            chrom_str=chrom_str,
            strand=strand,
            frag_len=frag_len,
            context_flank=context_flank,
            start_context=start_context,
            end_context=end_context,
            bed_lookup=bed_lookup,
            overlap=overlap,
            coverage_bw=coverage_bw,
            filters=filters,
            highqual_positions=highqual_positions,
        )

    raise ValueError(f"Invalid extract_type: {extract_type}")

def _extract_indels_from_read(
    read: pysam.AlignedSegment,
    bam_path: str,
    ref_cache: ReferenceCache,
    chrom_str: str,
    strand: str,
    frag_len: int,
    context_flank: int,
    start_context: str,
    end_context: str,
    bed_lookup: Optional[Dict[str, Tuple[List[int], List[int]]]] = None,
    coverage_bw: Optional["pyBigWig.bigWigFile"] = None,
) -> Tuple[Iterator[ReadFeatureRow], int]:
    """
    Extract insertion/deletion rows using match-style logic:
    - First collect all candidate indel positions cheaply.
    - Then lazily build rows via a generator.
    - Returns (iterator, n_opp) where n_opp = number of indels.
    """
    seq = read.query_sequence
    if not seq:
        return iter(()), 0

    seq_upper = seq.upper()
    
    if read.is_unmapped or not read.cigartuples:
        return iter(()), 0

    candidates: list[tuple[int, int, str]] = []  # (qpos, rpos, obs)

    ref_pos = read.reference_start
    query_pos = 0
    cigar = read.cigartuples or []
    
    for op, length in cigar:
        # M / = / X
        if op in (0, 7, 8):
            ref_pos += length
            query_pos += length

        # INSERTION (I)
        elif op == 1:
            obs_base = "I"
            indel_pos = ref_pos  # 1bp before insertion (0-based)
            pos_1based = indel_pos + 1

            # Fetch reference
            try:
                refb = ref_cache.fetch(chrom_str, pos_1based, pos_1based).upper()
            except Exception:
                refb = "N"

            candidates.append((query_pos, indel_pos, obs_base))

            query_pos += length  # consumes query

        # DELETION (D)
        elif op == 2:
            obs_base = "D"
            indel_pos = ref_pos  # deleted reference position
            pos_1based = indel_pos + 1

            try:
                refb = ref_cache.fetch(chrom_str, pos_1based, pos_1based).upper()
            except Exception:
                refb = "N"

            candidates.append((query_pos, indel_pos, obs_base))

            ref_pos += length  # consumes reference

        # SKIP (N)
        elif op == 3:
            ref_pos += length

        # SOFT/HARD CLIP (S/H)
        elif op in (4, 5):
            query_pos += length

    n_opp = len(candidates)
    if n_opp == 0:
        return iter(()), 0
        
    def _gen() -> Iterator[ReadFeatureRow]:
        for qpos, rpos, obs_base in candidates:
            pos_1based = rpos + 1

            try:
                refb = ref_cache.fetch(chrom_str, pos_1based, 0).upper()
            except Exception:
                refb = "N"

            row = _build_read_row(
                read=read,
                bam_path=bam_path,
                extract_type="indel",
                chrom=chrom_str,
                pos_1based=pos_1based,
                obs=obs_base,
                refb=refb,
                strand=strand,
                fragment_len=frag_len,
                query_pos=qpos,
                ref_cache=ref_cache,
                flank=context_flank,
                start_context=start_context,
                end_context=end_context,
                bed_lookup=bed_lookup,
                coverage_bw=coverage_bw,
            )
            if row is not None:
                yield row

    return _gen(), n_opp

def _extract_matches_from_read(
    read: pysam.AlignedSegment,
    bam_path: str,
    ref_cache: ReferenceCache,
    chrom_str: str,
    strand: str,
    frag_len: int,
    context_flank: int,
    start_context: str,
    end_context: str,
    seq_upper: str,
    bed_lookup: Optional[Dict[str, Tuple[List[int], List[int]]]] = None,
    overlap: Optional[Tuple[int, int]] = None,
    coverage_bw: Optional["pyBigWig.bigWigFile"] = None,
    filters: Optional["FilterContext"] = None,
    highqual_positions: Optional[set[int]] = None,
) -> Tuple[Iterator["ReadFeatureRow"], int]:
    """
    Extract one random matched base (for balancing) and count total match opportunities.
    Returns (iterator, n_opp), where n_opp = number of valid match positions.
    """
    # Determine overlap region
    start_ovl, end_ovl = overlap if overlap else (read.reference_start, read.reference_end)
    if start_ovl is None or end_ovl is None:
        return iter(()), 0

    candidates: List[Tuple[int, int, str]] = []

    for qpos, rpos, rbase in read.get_aligned_pairs(with_seq=True, matches_only=True):
        if rpos is None or rbase is None:
            continue
        if not (start_ovl <= rpos < end_ovl):
            continue
        if seq_upper[qpos] != rbase.upper():
            continue
        if highqual_positions is not None and rpos not in highqual_positions:
            continue
        if bed_lookup and not _in_bed_1based(chrom_str, rpos + 1, bed_lookup):
            continue
        if filters and _is_filtered(filters, chrom_str, rpos, "NA"):
            continue
        candidates.append((qpos, rpos, rbase))

    n_opp = len(candidates)
    if n_opp == 0:
        return iter(()), 0

    def _gen() -> Iterator["ReadFeatureRow"]:
        qpos, rpos, rbase = random.choice(candidates)
        pos_1based = rpos + 1
        row = _build_read_row(
            read=read,
            bam_path=bam_path,
            extract_type="match",
            chrom=chrom_str,
            pos_1based=pos_1based,
            obs=seq_upper[qpos],
            refb=rbase.upper(),
            strand=strand,
            fragment_len=frag_len,
            query_pos=qpos,
            ref_cache=ref_cache,
            flank=context_flank,
            start_context=start_context,
            end_context=end_context,
            bed_lookup=bed_lookup,
            coverage_bw=coverage_bw
        )
        if row is not None:
            yield row

    return _gen(), n_opp
            
def _extract_mismatches_from_read(
    read: pysam.AlignedSegment,
    bam_path: str,
    patient_id: str,
    ref_cache: ReferenceCache,
    chrom_str: str,
    strand: str,
    frag_len: int,
    context_flank: int,
    start_context: str,
    end_context: str,
    bed_lookup: Optional[Dict[str, Tuple[List[int], List[int]]]] = None,
    overlap: Optional[Tuple[int, int]] = None,
    coverage_bw: Optional["pyBigWig.bigWigFile"] = None,
    filters: Optional[FilterContext] = None,
    highqual_positions: Optional[set[int]] = None,
) -> Tuple[Iterator[ReadFeatureRow], int]:
    """
    Extract mismatch features for one read and count the number of
    valid (unfiltered, high-quality) positions in the overlap region.
    """
    seq = read.query_sequence
    if not seq:
        return iter(()), 0

    seq_upper = seq.upper()
    cigar = read.cigartuples or []
    md_tag_cache = read.get_tag("MD") if read.has_tag("MD") else ""
    other_err = count_other_errors_in_read_fast(md_tag_cache)
    n_ins, n_del, n_soft, n_hard = count_cigar_events(cigar)

    start_ovl, end_ovl = overlap if overlap else (read.reference_start, read.reference_end)
    if start_ovl is None or end_ovl is None:
        return iter(()), 0

    # Define valid positions in the overlap
    candidates = set(range(start_ovl, end_ovl))
    if highqual_positions:
        # Only convert to a set if needed and intersect efficiently
        if not isinstance(highqual_positions, set):
            highqual_positions = set(highqual_positions)

        # If the high-quality set is smaller, iterate over it instead of the big range
        if len(highqual_positions) < len(candidates):
            candidates = highqual_positions & candidates
        else:
            candidates &= highqual_positions

    # Apply BED/region filters
    if filters is not None:
        f = filters
        is_filt = _is_filtered
        candidates = {
            pos for pos in candidates
            if not is_filt(f, chrom_str, pos, patient_id)
        }
        
    n_opp = len(candidates)  # Number of eligible positions in overlap

    def _gen() -> Iterator[ReadFeatureRow]:
        for qpos, rpos, ref_base in _iter_mismatches_non_re(read):
            if rpos not in candidates:
                continue

            obs_base = seq_upper[qpos]
            refb = ref_base.upper()
            pos_1based = rpos + 1
            
            row = _build_read_row(
                read=read,
                bam_path=bam_path,
                extract_type="mismatch",
                chrom=chrom_str,
                pos_1based=pos_1based,
                obs=obs_base,
                refb=refb,
                strand=strand,
                fragment_len=frag_len,
                query_pos=qpos,
                ref_cache=ref_cache,
                flank=context_flank,
                start_context=start_context,
                end_context=end_context,
                n_insertions=n_ins,
                n_deletions=n_del,
                n_softclips=n_soft, 
                n_hardclips=n_hard,
                other_err=other_err,
                bed_lookup=bed_lookup,
                coverage_bw=coverage_bw,
            )
            if row is not None:
                yield row

    return _gen(), n_opp