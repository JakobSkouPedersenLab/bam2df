import gc
from pathlib import Path
import random
import re
import time
from typing import Optional

from src.features.read_ops import _extract_fragment_end_context, _extract_read_end_context, _fragment_is_proper_length, _paired_overlap, _mismatches_in_overlap
from src.features.filters import _high_quality_overlap, _high_quality_single_read, _load_bed, _read_filters, _make_interval_lookup, _read_has_mismatches
from src.features.extractors import extract_read_features
from src.features.reference import ReferenceCache
from src.features.constants import _PL_SCHEMA, _VALID_CHROMS
from src.utility.data import ChromStats, FilterContext, GenomeStats, ReadFeatureRow, rows_to_polars
from src.utility.sample import get_patient_id, get_patient_sample_id

import pysam
import polars as pl
import pyBigWig
import copy

class ChromosomeProcessor:
    """
    Orchestrates per-chromosome feature extraction from BAM files.
    Handles iteration, read pairing, filtering, and merging of Parquet parts.
    """
    def __init__(self,
                 config: dict,
                 verbose: bool = False,
                 filters: Optional[FilterContext] = None,
                 coverage_bw: Optional["pyBigWig.bigWigFile"] = None):
        
        self.config = config
        self.verbose = verbose
        self.filters = filters
        self.coverage_bw = coverage_bw

    # Public API
    def process(self, bam_path: str, fasta_path: str, extract_type: str,
                bed_path: str = None) -> tuple[pl.DataFrame, GenomeStats]:
        """
        Extracts features for all valid chromosomes in the BAM file,
        merges them into a single Parquet dataset, and optionally applies
        reservoir sampling for matches.
        """
        start_time = time.perf_counter()
        n_match_sample = self.config.get("n_match_sample", 0)
        n_proc = self.config.get("n_proc", 0)
        bed_dict = _load_bed(bed_path) if bed_path else {}
        
        dataframes: list[pl.DataFrame] = []
        genome_stats = GenomeStats()
        merged_df = None

        for chrom in _VALID_CHROMS:
            with pysam.AlignmentFile(bam_path, "rb") as bam:
                t0 = time.perf_counter()

                result = self._process_chromosome(
                    chrom=chrom,
                    bam_file=bam,
                    fasta_path=fasta_path,
                    extract_type=extract_type,
                    n_match_sample=n_match_sample,
                    bed_intervals=bed_dict.get(chrom, [])
                )
                
                elapsed = time.perf_counter() - t0
            
                if result:
                    rows, stats = result
                    df = rows_to_polars(rows, schema=_PL_SCHEMA)
                    dataframes.append(df)
                    genome_stats.add(stats)
                    
                    if self.verbose:
                        print(f"[ChromosomeProcessor:INFO] Finished {chrom} "
                            f"in {elapsed:.1f}s ({elapsed/60:.2f} min) — {stats}",
                            flush=True)
                
        if not dataframes:
            raise RuntimeError(f"No features extracted from BAM {bam_path}")
        
        merged_df = None
        for df in dataframes:
            if merged_df is None:
                merged_df = df
            else:
                merged_df = merged_df.extend(df)
    
        if self.verbose:
            elapsed = time.perf_counter() - start_time
            print(f"[ChromosomeProcessor:INFO] Combined {len(dataframes)} chroms in {elapsed/60:.2f} min",
                  flush=True
            )

        return merged_df, genome_stats

    # Internal helpers
    def _process_chromosome(
        self,
        chrom: str,
        bam_file: pysam.AlignmentFile,
        fasta_path: str,
        extract_type: str,
        n_match_sample: int,
        bed_intervals: list | None,
    ) -> Optional[tuple[list, ChromStats]]:
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
        
        cfg = self.config
        ctx_flank = cfg["context_flank"]
        patient_id = get_patient_id(bam_path)

        worker_reservoir: list[ReadFeatureRow] = []
        batch: list[ReadFeatureRow] = []
        coverage_bw = self.coverage_bw
        filters = self.filters
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
        
        mate_buffer = {}
        seen = 0
        
        for read in reads:
            # Read-level filtering
            if not _read_filters(read=read,
                                 extract_type=extract_type,
                                 min_mapq=min_mapq,
                                 max_nm=max_nm):
                stats.n_skipped += 1
                continue
            
            # Check if read contains mismatches
            if extract_type == "mismatch":
                if not _read_has_mismatches(read):
                    stats.n_skipped += 1
                    continue
            
            # If using paired-reads
            if use_paired_reads:
                qname = read.query_name
                if qname in mate_buffer:
                    mate = mate_buffer.pop(qname)
                    overlap = _paired_overlap(read, mate)
                    if not overlap:
                        continue
                    
                    # Record high quality positions and skip if none
                    highqual_overlap = _high_quality_overlap(read, mate, overlap, min_bq)
                    if not highqual_overlap:
                        stats.n_skipped += 1
                        continue
                    
                    # Skip pairs where the fragment is not proper length
                    min_len = cfg['min_len']
                    max_len = cfg['max_len']
                    if not _fragment_is_proper_length(read, mate, min_len, max_len):
                        stats.n_invalid_fragments += 1
                        stats.n_skipped += 1
                        continue
                    
                    # Skip read if is no mismatches in the overlapping region based on MD tag
                    if extract_type == "mismatch":
                        if not _mismatches_in_overlap(read, overlap, highqual_overlap):
                            stats.n_skipped += 1
                            continue
                        
                    # Record start/end context of read pairs
                    start_ctx, end_ctx = _extract_fragment_end_context(read, mate, k=ctx_flank)
                    
                    rows_iter, n_opp = extract_read_features(
                        read=read,
                        bam_path=bam_path,
                        patient_id=patient_id,
                        ref_cache=ref_cache,
                        extract_type=extract_type,
                        context_flank=ctx_flank,
                        start_context=start_ctx,
                        end_context=end_ctx,
                        bed_lookup=bed_lookup,
                        overlap=overlap,
                        coverage_bw=coverage_bw,
                        filters=filters,
                        highqual_positions=highqual_overlap
                    )

                    stats.n_total += n_opp
                    stats.n_reads += 1
                    
                    for row in rows_iter:
                        if extract_type == "mismatch":
                            stats.n_mismatches += 1
                        elif extract_type == "match":
                            stats.n_matches += 1
                        elif extract_type == "indel":
                            stats.n_indels += 1
                            
                        if extract_type == "match" and n_match_sample > 0:
                            seen += 1
                            if len(worker_reservoir) < n_match_sample:
                                worker_reservoir.append(row)
                            else:
                                j = random.randrange(seen)
                                if j < n_match_sample:
                                    worker_reservoir[j] = row
                        else:
                            batch.append(row)
                    
                    if stats.n_reads % 100_000 == 0 and self.verbose:
                        print(f"[ChromosomeProcessor:INFO] {stats}", flush=True)
                        break
                    
                else:
                    mate_buffer[qname] = read
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
                    extract_type=extract_type,
                    context_flank=ctx_flank,
                    start_context=start_ctx,
                    end_context=end_ctx,                
                    bed_lookup=bed_lookup,
                    overlap=None,
                    coverage_bw=coverage_bw,
                    filters=filters,
                    highqual_positions=highqual_positions,
                )

                stats.n_total += n_opp
                stats.n_reads += 1

                for row in rows_iter:
                    if extract_type == "mismatch":
                        stats.n_mismatches += 1
                    elif extract_type == "match":
                        stats.n_matches += 1
                    elif extract_type == "indel":
                        stats.n_indels += 1

                    if extract_type == "match" and n_match_sample > 0:
                        seen += 1
                        if len(worker_reservoir) < n_match_sample:
                            worker_reservoir.append(row)
                        else:
                            j = random.randrange(seen)
                            if j < n_match_sample:
                                worker_reservoir[j] = row
                    else:
                        batch.append(row)
                        
                if stats.n_reads % 100_000 == 0 and self.verbose:
                    print(f"[ChromosomeProcessor:INFO] {stats}", flush=True)
                
        if not batch and not worker_reservoir:
            return None
        
        all_rows = batch + worker_reservoir
        
        return all_rows, stats
    