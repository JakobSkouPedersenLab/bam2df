from pathlib import Path
from typing import Optional, Tuple
import polars as pl

from src.features.chr_processor import ChromosomeProcessor
from src.utility.sample import get_patient_id, get_sample_id
from src.utility.data import ChromStats, FilterContext, GenomeStats

import pyBigWig

class BamProcessor:
    """
    High-level orchestrator for processing one BAM file into model-ready data.
    """
    def __init__(self,
                 cfg: dict,
                 reference_path: str,
                 verbose: bool = False,
                 filters: Optional[FilterContext] = None,
                 replication_path: str = None,
                 mappability_path: str = None,
                 microsatellite_path: str = None,
                 repeatmasker_path: str = None,
                 chromatin_path: str = None,
                 bed_intervals: dict[str, list[tuple[int, int]]] = None,
                 n_workers: int = 1):
        
        self.cfg = cfg
        self.reference_path = reference_path
        self.verbose = verbose
        self.filters = filters
        self.replication_path = replication_path
        self.mappability_path = mappability_path
        self.microsatellite_path = microsatellite_path
        self.repeatmasker_path = repeatmasker_path
        self.chromatin_path = chromatin_path
        self.bed_intervals = bed_intervals
        self.n_workers = n_workers

    def extract_data(self, bam_path: str) -> Tuple[list[Path], pl.DataFrame]:
        """
        Extracts, filters, and summarizes data for one BAM file.
        """
        if self.verbose:
            print(f"[BamProcessor:INFO] Extracting data from {bam_path}", flush=True)

        chr_processor = ChromosomeProcessor(config=self.cfg,
                                            verbose=self.verbose,
                                            filters=self.filters,
                                            replication_path=self.replication_path,
                                            mappability_path=self.mappability_path,
                                            microsatellite_path=self.microsatellite_path,
                                            repeatmasker_path=self.repeatmasker_path,
                                            chromatin_path=self.chromatin_path,
                                            n_workers=self.n_workers)
        
        mutation_types = self.cfg.get("mutation_types")
        
        if not mutation_types:
            raise ValueError("cfg['mutation_types'] is missing or empty")
        
        chunk_paths, genome_stats = chr_processor.process(
            bam_path=bam_path,
            reference_path=self.reference_path,
            mutation_types=mutation_types,
            bed_intervals=self.bed_intervals,
        )
        
        if not chunk_paths:
            raise RuntimeError("No data extracted")
        
        # Collect stats
        info = self.summarize_info(genome_stats)
        
        if self.verbose:
            print(f"[BamProcessor:INFO] Completed BAM: {bam_path}", flush=True)

        return chunk_paths, info

    @classmethod
    def summarize_info(cls, genome_stats: GenomeStats) -> pl.DataFrame:
        rows = []

        # Define expected chromosomes (match your naming!)
        expected_chroms = [f"chr{i}" for i in range(1, 23)]

        for chrom in expected_chroms:
            stats = genome_stats.per_chrom.get(chrom)

            if stats is None:
                # Fill missing with zeros
                stats = ChromStats(chrom=chrom)

            rows.append(cls._metrics_from_counts(stats))

        # Total row (already aggregated)
        rows.append(cls._metrics_from_counts(genome_stats.total))

        return pl.DataFrame(rows)

    @staticmethod
    def _metrics_from_counts(stats: ChromStats) -> dict:
        """
        Compute counts a single ChromStats.
        """
        total_mismatches = int(stats.n_mismatches)
        total_matches = int(stats.n_matches)
        total_indels = int(stats.n_indels)
        total_coverage = int(stats.n_total)

        denom_cov = max(total_coverage, 1)  # avoid /0
        mismatch_load = total_mismatches / denom_cov
        indel_load = total_indels / denom_cov

        # opportunities excluding mismatches
        denom_beta = max(total_coverage - total_mismatches, 1)
        beta = total_matches / denom_beta

        return {
            "chrom": stats.chrom,
            "coverage": total_coverage,
            "mismatches": total_mismatches,
            "matches": total_matches,
            "indels": total_indels,
            "mismatch_load": mismatch_load,
            "indel_load": indel_load,
            "beta": beta,
        }