from itertools import chain
import random
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union
import pysam

from src.features.constants import OBS_DELETION, OBS_INSERTION, OBS_MATCH, OBS_SNV
from src.features.filters import _in_bed_1based
from src.utility.data import (FilterContext, ReferenceCache, count_other_errors_in_read)
from src.features.read import _build_read_row, _is_filtered, _iter_mismatches

import pyBigWig

def _fetch_ref_span_1based(ref_cache: ReferenceCache, chrom: str, start_1based: int, length: int) -> str:
    """Fetch reference bases for [start_1based, start_1based+length-1] using the cache's single-position API."""
    bases = []
    for p in range(start_1based, start_1based + length):
        try:
            b = ref_cache.fetch(chrom, p, 0).upper()
        except Exception:
            b = "N"
        bases.append(b)
    return "".join(bases)

def extract_read_features(
    read: pysam.AlignedSegment,
    bam_path: str,
    patient_id: str,
    ref_cache: ReferenceCache,
    mutation_types: List[str],
    context_flank: int,
    start_ctx: str,
    end_ctx: str,
    overlap: Optional[Tuple[int, int]] = None,
    bed_lookup: Optional[Dict[str, Tuple[List[int], List[int]]]] = None,
    coverage_bw: Optional["pyBigWig.bigWigFile"] = None,
    replication_bw: Optional["pyBigWig.bigWigFile"] = None,
    filters: Optional[FilterContext] = None,
    highqual_positions: Optional[set[int]] = None,
) -> Tuple[Iterator[dict], int]:
    """Yield one or more feature rows for the given read."""
    
    seq = read.query_sequence
    if not seq:
        return iter(()), 0
    
    seq_upper = seq.upper()
    chrom_str = read.reference_name or ""
    strand = "fwd" if not read.is_reverse else "rev"
    frag_len = abs(read.template_length)
    
    generators = []
    total_n_opp = 0
    
    for mutation_type in mutation_types:
        if mutation_type == "indel":
            gen, n_opp = _extract_indels_from_read(
                read=read,
                bam_path=bam_path,
                patient_id=patient_id,
                ref_cache=ref_cache,
                chrom=chrom_str,    
                strand=strand,
                frag_len=frag_len,
                context_flank=context_flank,
                start_ctx=start_ctx,
                end_ctx=end_ctx,
                bed_lookup=bed_lookup,
                coverage_bw=coverage_bw,
                replication_bw=replication_bw,
                filters=filters,
                highqual_positions=highqual_positions,
            )                                                                                                                                               
        elif mutation_type == "match":
            gen, n_opp = _extract_matches_from_read(
                read=read,
                bam_path=bam_path,
                patient_id=patient_id,
                ref_cache=ref_cache,
                chrom_str=chrom_str,
                strand=strand,
                frag_len=frag_len,
                context_flank=context_flank,
                start_ctx=start_ctx,
                end_ctx=end_ctx,
                bed_lookup=bed_lookup,
                seq_upper=seq_upper,
                coverage_bw=coverage_bw,
                replication_bw=replication_bw,
                filters=filters,
                highqual_positions=highqual_positions,
            )
        elif mutation_type == "mismatch":
            gen, n_opp = _extract_mismatches_from_read(
                read=read,
                bam_path=bam_path,
                patient_id=patient_id,
                ref_cache=ref_cache,
                chrom_str=chrom_str,
                strand=strand,
                frag_len=frag_len,
                context_flank=context_flank,
                start_ctx=start_ctx,
                end_ctx=end_ctx,
                bed_lookup=bed_lookup,
                overlap=overlap,
                coverage_bw=coverage_bw,
                replication_bw=replication_bw,
                filters=filters,
                highqual_positions=highqual_positions,
            )
        else:
            raise ValueError(f"Invalid mutation type: {mutation_type}")

        if n_opp > 0:
            generators.append(gen)
            total_n_opp += n_opp

    if not generators:
        return iter(()), 0

    return chain.from_iterable(generators), total_n_opp

def _extract_indels_from_read(
    read: pysam.AlignedSegment,
    bam_path: str,
    patient_id: str,
    ref_cache: ReferenceCache,
    chrom: str,
    strand: str,
    frag_len: int,
    context_flank: int,
    start_ctx: str,
    end_ctx: str,
    bed_lookup: Optional[Dict[str, Tuple[List[int], List[int]]]] = None,
    filters: Optional["FilterContext"] = None,
    highqual_positions: Optional[set[int]] = None,
    coverage_bw: Optional["pyBigWig.bigWigFile"] = None,
    replication_bw: Optional["pyBigWig.bigWigFile"] = None
) -> Tuple[Iterator[dict], int]:
    """Extract indels."""
    seq = read.query_sequence
    if not seq or read.is_unmapped or not read.cigartuples:
        return iter(()), 0

    # (qpos, rpos0, obs_type, obs_ref, obs_base)
    candidates: list[tuple[int, int, str, str, str]] = []

    ref_pos0 = read.reference_start  # 0-based
    query_pos = 0

    for op, length in read.cigartuples:
        if op in (0, 7, 8):  # M, =, X
            ref_pos0 += length
            query_pos += length

        elif op == 1:  # INSERTION
            anchor_rpos0 = ref_pos0 - 1
            if anchor_rpos0 < read.reference_start:
                query_pos += length
                continue

            ins = seq[query_pos: query_pos + length].upper()

            anchor_pos_1based = anchor_rpos0 + 1  # == ref_pos0
            try:
                anchor = ref_cache.fetch(chrom, anchor_pos_1based, 0).upper()
            except Exception:
                anchor = "N"

            obs_type = OBS_INSERTION
            obs_ref = anchor
            obs_base = anchor + ins

            candidates.append((query_pos, anchor_rpos0, obs_type, obs_ref, obs_base))
            query_pos += length

        elif op == 2:  # DELETION
            anchor_rpos0 = ref_pos0 - 1
            if anchor_rpos0 < read.reference_start:
                ref_pos0 += length
                continue

            anchor_pos_1based = anchor_rpos0 + 1  # == ref_pos0
            try:
                anchor = ref_cache.fetch(chrom, anchor_pos_1based, 0).upper()
            except Exception:
                anchor = "N"

            deleted = _fetch_ref_span_1based(ref_cache, chrom, anchor_pos_1based + 1, length)

            obs_type = OBS_DELETION
            obs_ref = anchor + deleted
            obs_base = anchor

            candidates.append((query_pos, anchor_rpos0, obs_type, obs_ref, obs_base))
            ref_pos0 += length
            
        elif op == 3:  # N
            ref_pos0 += length

        elif op == 4:  # S (soft clip consumes query)
            query_pos += length

        elif op == 5:  # H (hard clip consumes neither)
            pass

    if not candidates:
        return iter(()), 0

    # Apply high-quality filtering
    if highqual_positions:
        if not isinstance(highqual_positions, set):
            highqual_positions = set(highqual_positions)

        candidates = [
            (qpos, rpos0, obs_type, obs_ref, obs_base)
            for (qpos, rpos0, obs_type, obs_ref, obs_base) in candidates
            if rpos0 in highqual_positions
        ]

    # Apply region / BED filters
    if filters is not None:
        f = filters
        is_filt = _is_filtered
        candidates = [
            (qpos, rpos0, obs_type, obs_ref, obs_base)
            for (qpos, rpos0, obs_type, obs_ref, obs_base) in candidates
            if not is_filt(f, chrom, rpos0, patient_id)
        ]

    n_opp = len(candidates)
    if n_opp == 0:
        return iter(()), 0

    def _gen() -> Iterator[dict]:
        for qpos, rpos0, obs_type, obs_ref, obs_base in candidates:
            pos_1based = rpos0 + 1

            row = _build_read_row(
                read=read,
                bam_path=bam_path,
                chrom=chrom,
                pos_1based=pos_1based,
                obs_type=obs_type,
                obs_base=obs_base,
                obs_ref=obs_ref,
                strand=strand,
                fragment_len=frag_len,
                query_pos=qpos,
                ref_cache=ref_cache,
                flank=context_flank,
                start_ctx=start_ctx,
                end_ctx=end_ctx,
                bed_lookup=bed_lookup,
                coverage_bw=coverage_bw,
                replication_bw=replication_bw
            )
            if row is not None:
                yield row

    return _gen(), n_opp

def _extract_matches_from_read(
    read: pysam.AlignedSegment,
    bam_path: str,
    patient_id: str,
    ref_cache: ReferenceCache,
    chrom_str: str,
    strand: str,
    frag_len: int,
    context_flank: int,
    start_ctx: str,
    end_ctx: str,
    seq_upper: str,
    bed_lookup: Optional[Dict[str, Tuple[List[int], List[int]]]] = None,
    overlap: Optional[Tuple[int, int]] = None,
    coverage_bw: Optional["pyBigWig.bigWigFile"] = None,
    replication_bw: Optional["pyBigWig.bigWigFile"] = None,
    filters: Optional["FilterContext"] = None,
    highqual_positions: Optional[set[int]] = None,
) -> Tuple[Iterator[dict], int]:
    """
    Extract all matched bases that pass filters.
    Returns (iterator, n_opp), where n_opp = number of valid match positions.
    """
    start_ovl, end_ovl = overlap if overlap else (read.reference_start, read.reference_end)
    if start_ovl is None or end_ovl is None:
        return iter(()), 0

    # First pass: count opportunities
    n_opp = 0
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
        if filters and _is_filtered(filters, chrom_str, rpos, patient_id):
            continue
        n_opp += 1

    if n_opp == 0:
        return iter(()), 0

    # Second pass: yield rows
    def _gen() -> Iterator[dict]:
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
            if filters and _is_filtered(filters, chrom_str, rpos, patient_id):
                continue

            row = _build_read_row(
                read=read,
                bam_path=bam_path,
                chrom=chrom_str,
                pos_1based=rpos + 1,
                obs_type=OBS_MATCH,
                obs_base=seq_upper[qpos],
                obs_ref=rbase.upper(),
                strand=strand,
                fragment_len=frag_len,
                query_pos=qpos,
                ref_cache=ref_cache,
                flank=context_flank,
                start_ctx=start_ctx,
                end_ctx=end_ctx,
                bed_lookup=bed_lookup,
                coverage_bw=coverage_bw,
                replication_bw=replication_bw,
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
    start_ctx: str,
    end_ctx: str,
    bed_lookup: Optional[Dict[str, Tuple[List[int], List[int]]]] = None,
    overlap: Optional[Tuple[int, int]] = None,
    coverage_bw: Optional["pyBigWig.bigWigFile"] = None,
    replication_bw: Optional["pyBigWig.bigWigFile"] = None,
    filters: Optional[FilterContext] = None,
    highqual_positions: Optional[set[int]] = None,
) -> Tuple[Iterator[dict], int]:
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
    other_err = count_other_errors_in_read(md_tag_cache)

    start_ovl, end_ovl = overlap if overlap else (read.reference_start, read.reference_end)
    if start_ovl is None or end_ovl is None:
        return iter(()), 0
    
    use_hq = highqual_positions is not None
    if use_hq and not isinstance(highqual_positions, set):
        highqual_positions = set(highqual_positions)
        
    use_filters = filters is not None
    
    def eligible(rpos0: int) -> bool:
        if rpos0 < start_ovl or rpos0 >= end_ovl:
            return False
        if use_hq and (rpos0 not in highqual_positions):
            return False
        if use_filters and _is_filtered(filters, chrom_str, rpos0, patient_id):
            return False
        return True
    
    # Count opportunities
    if use_hq:
        if use_filters:
            n_opp = sum(
                1 for p in highqual_positions
                if start_ovl <= p < end_ovl and not _is_filtered(filters, chrom_str, p, patient_id)
            )
        else:
            n_opp = sum(1 for p in highqual_positions if start_ovl <= p < end_ovl)
    else:
        if use_filters:
            n_opp = sum(
                1 for p in range(start_ovl, end_ovl)
                if not _is_filtered(filters, chrom_str, p, patient_id)
            )
        else:
            n_opp = end_ovl - start_ovl
    
    def _gen() -> Iterator[dict]:
        for qpos, rpos0, ref_base in _iter_mismatches(read):
            if not eligible(rpos0):
                continue

            obs_base = seq_upper[qpos]
            obs_ref = ref_base.upper()
            pos_1based = rpos0 + 1

            row = _build_read_row(
                read=read,
                bam_path=bam_path,
                chrom=chrom_str,
                pos_1based=pos_1based,
                obs_type=OBS_SNV,
                obs_base=obs_base,
                obs_ref=obs_ref,
                strand=strand,
                fragment_len=frag_len,
                query_pos=qpos,
                ref_cache=ref_cache,
                flank=context_flank,
                start_ctx=start_ctx,
                end_ctx=end_ctx,
                other_err=other_err,
                bed_lookup=bed_lookup,
                coverage_bw=coverage_bw,
                replication_bw=replication_bw,
            )
            if row is not None:
                yield row

    return _gen(), n_opp