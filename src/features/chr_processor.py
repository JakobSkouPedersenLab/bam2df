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

from src.features.read import (_extract_fragment_end_context, _extract_read_end_context, _fragment_is_proper_length,
                               _paired_overlap, indel_in_overlap, mismatch_in_overlap, match_in_overlap)
from src.features.chromatin import extract_chromatin_features
from src.features.filters import (_high_quality_overlap, _high_quality_single_read, _load_bed, _read_filters, _make_interval_lookup,
                                  read_has_indel, read_has_mismatch, read_has_match, read_matches_any_type,
                                  read_matches_any_type_in_overlap)
from src.features.extractors import extract_read_features
from src.features.constants import (_CHROMATIN_SCHEMA, _PL_SCHEMA, _REPLICATION_TIMING_SCHEMA, _VALID_CHROMS,
                                    OBS_DELETION, OBS_INSERTION, OBS_MATCH, OBS_SNV)
from src.utility.data import ChromStats, FilterContext, GenomeStats, ReferenceCache, get_open_indices, rows_to_polars
from src.utility.sample import get_patient_id, get_sample_id
from src.utility.io import get_hpc_tmp_dir, load_chr_mask

def process_chromosome(
    chrom: str,
    bam_file: pysam.AlignmentFile,
    fasta_path: str,
    mutation_types: list[str],
    n_match_sample: int,
    tmp_dir: Path,
    config: dict,
    verbose: bool,
    bed_intervals: list | None,
    filters: FilterContext | None,
    coverage_bw: pyBigWig.pyBigWig | None,
    replication_bw: pyBigWig.pyBigWig | None
) -> Optional[tuple[list[dict], ChromStats]]:
    """
    Process one chromosome and return the output file path and ChromStats.
    """
    bed_lookup = _make_interval_lookup({chrom: bed_intervals or []})
    ref_cache = ReferenceCache(fasta_path)
    bam_path = (
        bam_file.filename.decode()
        if isinstance(bam_file.filename, bytes)
        else bam_file.filename
    )
    
    cfg = config
    ctx_flank = cfg["context_flank"]
    patient_id = get_patient_id(bam_path)
    sample_id = get_sample_id(bam_path)
    
    out_path = tmp_dir / f"{sample_id}_{chrom}.parquet"
    
    # Output bookkeeping
    chunk_paths: list[Path] = []
    chunk_idx = 0

    # In-memory buffers (bounded)
    rows_chunk: list[dict] = []
    worker_reservoir: list[dict] = []
    seen = 0
    
    stats = ChromStats(chrom=chrom)
    
    # Set up read iterator
    if bed_intervals:
        iters = (bam_file.fetch(chrom, s, e) for s, e in bed_intervals)
        reads = (r for sub in iters for r in sub)
    else:
        reads = bam_file.fetch(chrom)
    
    min_bq = cfg.get("min_base_quality", 0)
    min_mapq = cfg.get("min_mapq", 0)
    max_nm = cfg.get("max_nm", 2)
    use_paired_reads = cfg.get("use_paired_reads", True)
    chunk_size = cfg.get("chunk_size", 50000)
    
    mate_buffer = {}
    
    # Set up schema
    schema = dict(_PL_SCHEMA)
    if replication_bw is not None:
        schema.update(_REPLICATION_TIMING_SCHEMA)
        
    def flush_chunk(schema) -> None:
        """Write current chunk to a new Parquet file and clear the buffer."""
        nonlocal chunk_idx, rows_chunk

        if not rows_chunk:
            return

        chunk_path = tmp_dir / f"{sample_id}_{chrom}_chunk{chunk_idx:05d}.parquet"
        df = rows_to_polars(rows_chunk, schema=schema)
        df.write_parquet(chunk_path)

        chunk_paths.append(chunk_path)
        chunk_idx += 1
        rows_chunk.clear()
        
    for read in reads:
        # Apply read filters
        if not _read_filters(read=read,
                             min_mapq=min_mapq,
                             max_nm=max_nm):
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

            max_mate_buffer = cfg.get("max_mate_buffer", 1_000_000)
            if len(mate_buffer) >= max_mate_buffer:
                mate_buffer.clear()
                stats.n_skipped += 1
                continue

            qname = read.query_name

            mate = mate_buffer.pop(qname, None)
            if mate is None:
                mate_buffer[qname] = read  # buffer read1 OR read2
                continue

            # Validate we got one read1 and one read2
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

            highqual_overlap = _high_quality_overlap(read2, read1, overlap, min_bq)
            if not highqual_overlap:
                stats.n_skipped += 1
                continue

            min_len = cfg["min_len"]
            max_len = cfg["max_len"]
            if not _fragment_is_proper_length(read1, read2, min_len, max_len):
                stats.n_invalid_fragments += 1
                stats.n_skipped += 1
                continue

            overlap_predicates = {
                "match": match_in_overlap,
                "mismatch": mismatch_in_overlap,
                "indel": indel_in_overlap,
            }
            if not read_matches_any_type_in_overlap(read2, overlap, highqual_overlap,
                                                    mutation_types, overlap_predicates):
                stats.n_skipped += 1
                continue

            start_ctx, end_ctx = _extract_fragment_end_context(read1, read2, k=ctx_flank)

            # driver read: keep as read2
            driver = read2
            rows_iter, n_opp = extract_read_features(
                read=driver,
                bam_path=bam_path,
                patient_id=patient_id,
                ref_cache=ref_cache,
                mutation_types=mutation_types,
                context_flank=ctx_flank,
                start_ctx=start_ctx,
                end_ctx=end_ctx,
                bed_lookup=bed_lookup,
                overlap=overlap,
                coverage_bw=coverage_bw,
                replication_bw=replication_bw,
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
            # Calculate high quality positions
            highqual_positions = _high_quality_single_read(read, min_bq)
            
            # Calculate read context
            start_ctx, end_ctx = _extract_read_end_context(read, k=ctx_flank)
            
            # Treat each read independently
            rows_iter, n_opp = extract_read_features(
                read=read,
                bam_path=bam_path,
                patient_id=patient_id,
                ref_cache=ref_cache,
                mutation_types=mutation_types,
                context_flank=ctx_flank,
                start_ctx=start_ctx,
                end_context=end_ctx,                
                bed_lookup=bed_lookup,
                overlap=None,
                coverage_bw=coverage_bw,
                replication_bw=replication_bw,
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
        chunk_path = tmp_dir / f"{sample_id}_{chrom}_reservoir.parquet"
        df = rows_to_polars(worker_reservoir, schema=_PL_SCHEMA)
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
    bed_intervals: Optional[list[tuple[int, int]]],
    tmp_dir: Path,
    config: dict,
    filters: Optional[FilterContext] = None,
    coverage_path: str = None,
    replication_path: str = None,
    chromatin_path: str = None,
    verbose: bool = False,
) -> Optional[tuple[list[Path], ChromStats]]:
    """
    Worker entry point for per-chromosome feature extraction.

    This function is executed in a separate process. It:
    1. Stream read-level features from BAM → Parquet (via process_chromosome)
    2. Optionally annotate chromatin features (per chromosome)
    3. Return (chrom, parquet_path, ChromStats)
    """
    t0 = time.perf_counter()
    
    coverage_bw = pyBigWig.open(coverage_path) if coverage_path else None
    replication_bw = pyBigWig.open(replication_path) if replication_path else None

    with pysam.AlignmentFile(bam_path, "rb") as bam:
        try:
            result = process_chromosome(
                chrom=chrom,
                bam_file=bam,
                fasta_path=fasta_path,
                mutation_types=mutation_types,
                n_match_sample=n_match_sample,
                tmp_dir=tmp_dir,
                bed_intervals=bed_intervals,
                config=config,
                filters=filters,
                coverage_bw=coverage_bw,
                replication_bw=replication_bw,
                verbose=verbose
            )
        finally:
            if coverage_bw is not None:
                coverage_bw.close()
            if replication_bw is not None:
                replication_bw.close()

    if not result:
        return None

    chunk_paths, stats = result

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

        for parquet_path in chunk_paths:
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
            f"in {elapsed:.1f}s ({elapsed/60:.2f} min) — {stats}",
            flush=True,
        )

    return chunk_paths, stats

class ChromosomeProcessor:
    """
    Orchestrates per-chromosome feature extraction.
    """
    def __init__(self,
                 config: dict,
                 verbose: bool = False,
                 filters: Optional[FilterContext] = None,
                 coverage_path: str = None,
                 replication_path: str = None,
                 chromatin_path: str = None,
                 n_workers: int = 1):
        
        self.config = config
        self.verbose = verbose
        self.filters = filters
        self.coverage_path = coverage_path
        self.replication_path = replication_path
        self.chromatin_path = chromatin_path
        self.n_workers = n_workers
        self.tmp_dir = get_hpc_tmp_dir(prefix="chrproc")
    
    # Public API
    def process(self,
                bam_path: str,
                reference_path: str,
                mutation_types: list[str],
                bed_path: str = None
    ) -> tuple[pl.LazyFrame, GenomeStats]:

        start_time = time.perf_counter()
        n_match_sample = self.config.get("n_match_sample", 0)
        bed_dict = _load_bed(bed_path) if bed_path else {}

        chunk_paths_all: list[Path] = []
        genome_stats = GenomeStats()

        if self.verbose:
            print(f"[ChromosomeProcessor:INFO] Using tmp dir: {self.tmp_dir}", flush=True)
        
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=self.n_workers, mp_context=ctx) as executor:
            futures = []

            for chrom in _VALID_CHROMS:
                futures.append(
                    executor.submit(
                        _process_chromosome_worker,
                        chrom=chrom,
                        bam_path=bam_path,
                        fasta_path=reference_path,
                        mutation_types=mutation_types,
                        n_match_sample=n_match_sample,
                        bed_intervals=bed_dict.get(chrom, []),
                        tmp_dir=self.tmp_dir,
                        chromatin_path=self.chromatin_path,
                        config=self.config,
                        filters=self.filters,
                        coverage_path=self.coverage_path,
                        replication_path=self.replication_path,
                        verbose=self.verbose,
                    )
                )

            for fut in futures:
                result = fut.result()
                if result is None:
                    continue

                chunk_paths, stats = result
                chunk_paths_all.extend(chunk_paths)
                genome_stats.add(stats)

        if not chunk_paths_all:
            raise RuntimeError(f"No features extracted from BAM {bam_path}")

        # Build lazy frames per chunk, optionally hstack chromatin sidecar
        per_chunk_lfs: list[pl.LazyFrame] = []
        use_chromatin = self.chromatin_path is not None

        for chunk_path in chunk_paths_all:
            lf_chunk = pl.scan_parquet(chunk_path)

            if use_chromatin:
                chrom_path = chunk_path.with_suffix(".chrom.parquet")
                if chrom_path.exists():
                    lf_chrom = pl.scan_parquet(chrom_path)
                    lf_chunk = pl.concat([lf_chunk, lf_chrom], how="horizontal")

            per_chunk_lfs.append(lf_chunk)

        # Vertically concat all chunks
        lf = pl.concat(per_chunk_lfs, how="vertical")

        # Optional: global sorting
        chr_order = {c: i for i, c in enumerate(_VALID_CHROMS)}
        lf = (
            lf.with_columns(
                pl.col("chr").replace(chr_order).cast(pl.Int32).alias("_chr_order")
            )
            .sort(["_chr_order", "genomic_pos"])
            .drop("_chr_order")
        )

        if self.verbose:
            elapsed = time.perf_counter() - start_time
            print(
                f"[ChromosomeProcessor:INFO] Prepared lazy plan from {len(chunk_paths_all)} chunks "
                f"in {elapsed/60:.2f} min",
                flush=True,
            )

        return lf, genome_stats
    