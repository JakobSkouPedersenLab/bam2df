import gc
import itertools
from pathlib import Path
import random
import re
import time
from typing import Optional
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
import pysam
import polars as pl
import pyBigWig
import gc
from collections import deque

from src.features.read import (_calculate_fragment_length, _extract_fragment_end_context, _extract_read_end_context, _fragment_is_proper_length,
                               _paired_overlap, indel_in_overlap, mismatch_in_overlap, match_in_overlap)
from src.features.chromatin import extract_chromatin_features
from src.features.filters import (_high_quality_overlap, _high_quality_single_read, _load_bed, _read_filters, _make_interval_lookup,
                                  read_has_indel, read_has_mismatch, read_has_match, read_matches_any_type,
                                  read_matches_any_type_in_overlap)
from src.features.extractors import extract_read_features
from src.features.constants import (_CHROMATIN_SCHEMA, _MAPPABILITY_SCHEMA, _MICROSAT_SCHEMA, _PL_SCHEMA, _REPLICATION_TIMING_SCHEMA, _REPMASK_SCHEMA, _VALID_CHROMS,
                                    OBS_DELETION, OBS_INSERTION, OBS_MATCH, OBS_SNV)
from src.utility.data import ChromStats, FilterContext, GenomeStats, MicrosatelliteLookup, ReferenceCache, RepeatMaskerLookup, get_open_indices, rows_to_polars
from src.utility.io import get_hpc_tmp_dir, load_chr_mask, load_microsatellites_for_chrom, load_repeatmasker_for_chrom

def process_chromosome(
    chrom: str,
    start: int,
    end: int,
    bam_file: pysam.AlignmentFile,
    fasta_path: str,
    mutation_types: list[str],
    n_match_sample: int,
    tmp_dir: Path,
    config: dict,
    verbose: bool,
    filters: FilterContext | None,
    replication_bw: pyBigWig.pyBigWig | None,
    mappability_bw: pyBigWig.pyBigWig | None,
    microsatellite_lookup: MicrosatelliteLookup | None,
    repeatmasker_lookup: RepeatMaskerLookup | None,
    patient_id: str,
    sample_id: str,
) -> Optional[tuple[list[Path], ChromStats]]:
    """
    Process one chromosome interval [start, end) and return chunk paths + ChromStats.
    """
    if end <= start:
        return None
    
    ref_cache = ReferenceCache(fasta_path)
    bam_path = (
        bam_file.filename.decode()
        if isinstance(bam_file.filename, bytes)
        else bam_file.filename
    )
    
    cfg = config
    ctx_flank = cfg["context_flank"]
    
    chunk_paths: list[Path] = []
    chunk_idx = 0

    rows_chunk: list[dict] = []
    worker_reservoir: list[dict] = []
    seen = 0
    
    stats = ChromStats(chrom=chrom)
    
    min_mapq = cfg.get("min_mapq", 60)
    min_bq = cfg["min_base_quality"]  # validated by validate_bam_config
    chunk_size = cfg.get("chunk_size", 100000)
    use_paired_reads = cfg.get("use_paired_reads", True)
    
    mate_buffer = {}
    mate_fifo = deque()
    max_mate_buffer = cfg.get("max_mate_buffer", 1_000_000)

    schema = dict(_PL_SCHEMA)
    if replication_bw is not None:
        schema.update(_REPLICATION_TIMING_SCHEMA)
    if mappability_bw is not None:
        schema.update(_MAPPABILITY_SCHEMA)
    if microsatellite_lookup is not None:
        schema.update(_MICROSAT_SCHEMA)
    if repeatmasker_lookup is not None:
        schema.update(_REPMASK_SCHEMA)
        
    def ensure_capacity():
        while len(mate_buffer) >= max_mate_buffer:
            if not mate_fifo:
                stats.n_evicted += len(mate_buffer)
                mate_buffer.clear()
                break

            old_qname = mate_fifo.popleft()
            if mate_buffer.pop(old_qname, None) is not None:
                stats.n_evicted += 1

    def flush_chunk(schema) -> None:
        nonlocal chunk_idx, rows_chunk

        if not rows_chunk:
            return

        chunk_path = tmp_dir / (
            f"{sample_id}_{chrom}_{start}_{end}_chunk{chunk_idx:05d}.parquet"
        )
        df = rows_to_polars(rows_chunk, schema=schema)
        df.write_parquet(chunk_path, compression="zstd")

        chunk_paths.append(chunk_path)
        chunk_idx += 1
        rows_chunk.clear()

    def read_overlaps_interval(read: pysam.AlignedSegment, region_start: int, region_end: int) -> bool:
        """
        Fast coarse filter: keep only reads with any reference overlap with [region_start, region_end).
        """
        read_start = read.reference_start
        read_end = read.reference_end
        if read_start is None or read_end is None:
            return False
        return read_start < region_end and read_end > region_start

    # Interval-bounded fetch
    reads = bam_file.fetch(chrom, start, end)
    
    # Iterate reads
    for read in reads:
        # Apply read filters
        if not _read_filters(
            read=read,
            min_mapq=min_mapq,
        ):
            stats.n_skipped += 1
            continue
            
        # Filter read on match/mismatch/indel
        read_predicates = {
            "match":    read_has_match,
            "mismatch": read_has_mismatch,
            "indel":    read_has_indel,
        }
        if not read_matches_any_type(read, mutation_types, read_predicates):
            stats.n_skipped += 1
            continue
                
        # If using paired-reads
        if use_paired_reads:
            if read.is_secondary or read.is_supplementary:
                stats.n_skipped += 1
                continue

            qname = read.query_name

            mate = mate_buffer.pop(qname, None)
            if mate is None:
                ensure_capacity()
                mate_buffer[qname] = read
                mate_fifo.append(qname)
                continue

            if read.is_read1 and mate.is_read2:
                read1, read2 = read, mate
            elif read.is_read2 and mate.is_read1:
                read1, read2 = mate, read
            else:
                stats.n_skipped += 1
                continue

            if read1.is_unmapped or read2.is_unmapped:
                stats.n_skipped += 1
                continue

            overlap = _paired_overlap(read1, read2)
            if not overlap:
                stats.n_skipped += 1
                continue
                
            # Restrict pair overlap to BED interval
            ovl_start, ovl_end = overlap
            clipped_overlap = (max(ovl_start, start), min(ovl_end, end))
            if clipped_overlap[1] <= clipped_overlap[0]:
                stats.n_skipped += 1
                continue

            highqual_overlap = _high_quality_overlap(read1, read2, clipped_overlap, min_bq)
            if not highqual_overlap:
                stats.n_skipped += 1
                continue

            min_len = cfg["min_len"]
            max_len = cfg["max_len"]
            if not _fragment_is_proper_length(read1, read2, min_len, max_len):
                stats.n_invalid_fragments += 1
                stats.n_skipped += 1
                continue
            
            frag_len = _calculate_fragment_length(read1, read2)
            
            start_ctx, end_ctx = _extract_fragment_end_context(read1, read2, k=ctx_flank)

            # driver read: keep as read2
            driver = read2
            rows_iter, n_opp = extract_read_features(
                read=driver,
                bam_path=bam_path,
                patient_id=patient_id,
                ref_cache=ref_cache,
                mutation_types=mutation_types,
                frag_len=frag_len,
                min_len=min_len,
                max_len=max_len,
                min_mapq=min_mapq,
                min_bq=min_bq,
                context_flank=ctx_flank,
                start_ctx=start_ctx,
                end_ctx=end_ctx,
                overlap=clipped_overlap,
                replication_bw=replication_bw,
                mappability_bw=mappability_bw,
                microsatellite_lookup=microsatellite_lookup,
                repeatmasker_lookup=repeatmasker_lookup,
                filters=filters,
                highqual_positions=highqual_overlap,
            )

            stats.n_total += n_opp
            stats.n_reads += 1
                
            for row in rows_iter:
                t = row["obs_type"]
                if t == OBS_SNV:
                    stats.n_mismatches += 1
                elif t == OBS_MATCH:
                    stats.n_matches += 1
                elif t in [OBS_INSERTION, OBS_DELETION]:
                    stats.n_indels += 1
                    
                if t == OBS_MATCH and n_match_sample > 0:
                    seen += 1
                    if len(worker_reservoir) < n_match_sample:
                        worker_reservoir.append(row)
                    else:
                        j = random.randrange(seen)
                        if j < n_match_sample:
                            worker_reservoir[j] = row
                else:
                    rows_chunk.append(row)
                    
                if len(rows_chunk) >= chunk_size:
                    flush_chunk(schema)
        else:
            highqual_positions = _high_quality_single_read(read, min_bq)
            frag_len = read.template_length
            start_ctx, end_ctx = _extract_read_end_context(read, k=ctx_flank)
            
            # Check fragment is proper length
            min_len = cfg["min_len"]
            max_len = cfg["max_len"]
            if frag_len < min_len or frag_len > max_len:
                stats.n_invalid_fragments += 1
                stats.n_skipped += 1
                continue
            
            # Treat each read independently
            rows_iter, n_opp = extract_read_features(
                read=read,
                bam_path=bam_path,
                patient_id=patient_id,
                ref_cache=ref_cache,
                mutation_types=mutation_types,
                frag_len=frag_len,
                min_len=min_len,
                max_len=max_len,
                min_mapq=min_mapq,
                min_bq=min_bq,
                context_flank=ctx_flank,
                start_ctx=start_ctx,
                end_ctx=end_ctx,
                overlap=None,
                replication_bw=replication_bw,
                mappability_bw=mappability_bw,
                microsatellite_lookup=microsatellite_lookup,
                repeatmasker_lookup=repeatmasker_lookup,
                filters=filters,
                highqual_positions=highqual_positions,
            )

            stats.n_total += n_opp
            stats.n_reads += 1

            for row in rows_iter:
                t = row["obs_type"]
                if t == OBS_SNV:
                    stats.n_mismatches += 1
                elif t == OBS_MATCH:
                    stats.n_matches += 1
                elif t in [OBS_INSERTION, OBS_DELETION]:
                        stats.n_indels += 1
                    
                if t == OBS_MATCH and n_match_sample > 0:
                    seen += 1
                    if len(worker_reservoir) < n_match_sample:
                        worker_reservoir.append(row)
                    else:
                        j = random.randrange(seen)
                        if j < n_match_sample:
                            worker_reservoir[j] = row
                else:
                    rows_chunk.append(row)
                    
                if len(rows_chunk) >= chunk_size:
                    flush_chunk(schema)
                
    # Final flush
    flush_chunk(schema)
    
    # Write reservoir at the end
    if worker_reservoir:
        chunk_path = tmp_dir / f"{sample_id}_{chrom}_{start}_{end}_reservoir.parquet"
        df = rows_to_polars(worker_reservoir, schema=schema)
        df.write_parquet(chunk_path, compression="zstd")
        chunk_paths.append(chunk_path)

    if not chunk_paths:
        return None

    return chunk_paths, stats

def _process_chromosome_worker(
    chrom: str,
    bam_path: str,
    fasta_path: str,
    mutation_types: list[str],
    n_match_sample: int,
    tmp_dir: Path,
    config: dict,
    filters: Optional[FilterContext] = None,
    replication_path: str = None,
    mappability_path: str = None,
    microsatellite_path: str = None,
    repeatmasker_path: str = None,
    chromatin_path: str = None,
    verbose: bool = False,
    bed_intervals: list[tuple[int, int]] = None,
    patient_id: str = None,
    sample_id: str = None,
) -> Optional[tuple[list[Path], ChromStats]]:
    """
    Worker entry point for per-chromosome feature extraction.
    """
    t0 = time.perf_counter()
    
    if bed_intervals is not None and len(bed_intervals) == 0:
        if verbose:
            print(
                f"[ChromosomeProcessor:INFO] Skipping {chrom}: empty BED interval list",
                flush=True,
            )
        return None
    
    replication_bw = pyBigWig.open(replication_path) if replication_path else None
    mappability_bw = pyBigWig.open(mappability_path) if mappability_path else None
    microsatellite_lookup = (
        load_microsatellites_for_chrom(microsatellite_path, chrom)
        if microsatellite_path
        else None
    )
    repeatmasker_lookup = (
        load_repeatmasker_for_chrom(repeatmasker_path, chrom)
        if repeatmasker_path
        else None
    )
    
    all_chunk_paths: list[Path] = []
    chrom_stats_total = ChromStats(chrom=chrom)

    try:
        with pysam.AlignmentFile(bam_path, "rb") as bam:
            if bed_intervals is None:
                intervals_to_process = [(0, bam.get_reference_length(chrom))]
            else:
                intervals_to_process = bed_intervals
                
            for interval_idx, (start, end) in enumerate(intervals_to_process, start=1):
                if end <= start:
                    if verbose:
                        print(
                            f"[ChromosomeProcessor:WARN] Skipping invalid interval "
                            f"{chrom}:{start}-{end}",
                            flush=True,
                        )
                    continue
                    
                result = process_chromosome(
                    chrom=chrom,
                    start=start,
                    end=end,
                    bam_file=bam,
                    fasta_path=fasta_path,
                    mutation_types=mutation_types,
                    n_match_sample=n_match_sample,
                    tmp_dir=tmp_dir,
                    config=config,
                    filters=filters,
                    replication_bw=replication_bw,
                    mappability_bw=mappability_bw,
                    microsatellite_lookup=microsatellite_lookup,
                    repeatmasker_lookup=repeatmasker_lookup,
                    patient_id=patient_id,
                    sample_id=sample_id,
                    verbose=verbose,
                ) 
                
                if not result:
                    continue
                
                chunk_paths, stats = result
                all_chunk_paths.extend(chunk_paths)
                chrom_stats_total += stats
            
    finally:
        if replication_bw is not None:
            replication_bw.close()
        if mappability_bw is not None:
            mappability_bw.close()

    if not all_chunk_paths:
        return None

    # Chromatin annotation (optional)
    if chromatin_path is not None:
        bin_size_bp = config["bin_size_bp"]
        chrom_root = Path(chromatin_path)

        # Load only this chromosome masks (ATAC + DNase)
        mask_atac = load_chr_mask(chrom_root, "ATAC", chrom)
        mask_dnase = load_chr_mask(chrom_root, "DNase", chrom)

        open_idx_by_mod = {
            "ATAC": get_open_indices(mask_atac),
            "DNase": get_open_indices(mask_dnase),
        }

        # Free sparse matrices early; keep only open indices arrays
        del mask_atac, mask_dnase
        gc.collect()

        for parquet_path in all_chunk_paths:
            # only genomic_pos -> bins
            bins = (
                pl.read_parquet(parquet_path, columns=["genomic_pos"])
                .with_columns((pl.col("genomic_pos") // bin_size_bp).cast(pl.Int64).alias("bin_idx"))
                .get_column("bin_idx")
                .to_numpy()
            )

            atac = extract_chromatin_features(
                bins=bins, open_idx=open_idx_by_mod["ATAC"],
                modality="ATAC", bin_size_bp=bin_size_bp
            )
            dnase = extract_chromatin_features(
                bins=bins, open_idx=open_idx_by_mod["DNase"],
                modality="DNase", bin_size_bp=bin_size_bp
            )

            feat_df = pl.DataFrame({**atac, **dnase}, schema=_CHROMATIN_SCHEMA)
            feat_path = parquet_path.with_suffix(".chrom.parquet")
            feat_df.write_parquet(feat_path, compression="zstd")

            del bins, feat_df
            gc.collect()

    elapsed = time.perf_counter() - t0

    if verbose:
        print(
            f"[ChromosomeProcessor:INFO] Finished {chrom} "
            f"in {elapsed:.1f}s ({elapsed/60:.2f} min) — {chrom_stats_total}",
            flush=True,
        )

    return all_chunk_paths, chrom_stats_total

class ChromosomeProcessor:
    """
    Orchestrates per-chromosome feature extraction.
    """
    def __init__(self,
                 config: dict,
                 verbose: bool = False,
                 filters: Optional[FilterContext] = None,
                 replication_path: str = None,
                 mappability_path: str = None,
                 microsatellite_path: str = None,
                 repeatmasker_path: str = None,
                 chromatin_path: str = None,
                 n_workers: int = 1):
        
        self.config = config
        self.verbose = verbose
        self.filters = filters
        self.replication_path = replication_path
        self.mappability_path = mappability_path
        self.microsatellite_path = microsatellite_path
        self.repeatmasker_path = repeatmasker_path
        self.chromatin_path = chromatin_path
        self.n_workers = n_workers
        self.tmp_dir = get_hpc_tmp_dir(prefix="chrproc")
    
    # Public API
    def process(self,
                bam_path: str,
                reference_path: str,
                mutation_types: list[str],
                bed_intervals: dict[str, list[tuple[int, int]]] = None,
                patient_id: str = None,
                sample_id: str = None,
    ) -> tuple[pl.LazyFrame, GenomeStats]:

        start_time = time.perf_counter()
        n_match_sample = self.config.get("n_match_sample", 0)

        chunk_paths_all: list[Path] = []
        genome_stats = GenomeStats()

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=self.n_workers, mp_context=ctx) as executor:
            futures = []

            for chrom in _VALID_CHROMS:
                chrom_bed_intervals = None
                if bed_intervals is not None:
                    chrom_bed_intervals = bed_intervals.get(chrom, [])
                    if not chrom_bed_intervals:
                        continue
                
                futures.append(
                    executor.submit(
                        _process_chromosome_worker,
                        chrom=chrom,
                        bam_path=bam_path,
                        fasta_path=reference_path,
                        mutation_types=mutation_types,
                        n_match_sample=n_match_sample,
                        tmp_dir=self.tmp_dir,
                        chromatin_path=self.chromatin_path,
                        config=self.config,
                        filters=self.filters.for_chromosome(chrom) if self.filters is not None else None,
                        replication_path=self.replication_path,
                        mappability_path=self.mappability_path,
                        microsatellite_path=self.microsatellite_path,
                        repeatmasker_path=self.repeatmasker_path,
                        verbose=self.verbose,
                        bed_intervals=chrom_bed_intervals,
                        patient_id=patient_id,
                        sample_id=sample_id,
                    )
                )

            for fut in futures:
                result = fut.result()
                if result is None:
                    continue

                chunk_paths, stats = result
                chunk_paths_all.extend(chunk_paths)
                genome_stats += stats

        if not chunk_paths_all:
            raise RuntimeError(f"No features extracted from BAM {bam_path}")

        if self.verbose:
            elapsed = time.perf_counter() - start_time
            print(
                f"[ChromosomeProcessor:INFO] Prepared {len(chunk_paths_all)} chunk files "
                f"in {elapsed/60:.2f} min",
                flush=True,
            )

        return chunk_paths_all, genome_stats
    