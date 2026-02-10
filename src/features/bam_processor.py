from typing import Optional, Tuple
import polars as pl

from src.features.chr_processor import ChromosomeProcessor
from src.utility.sample import get_patient_id, get_sample_id
from src.utility.data import ChromStats, FilterContext

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
                 coverage_path: str = None,
                 replication_path: str = None,
                 chromatin_path: str = None,
                 n_workers: int = 1):
        
        self.cfg = cfg
        self.reference_path = reference_path
        self.verbose = verbose
        self.filters = filters
        self.coverage_path = coverage_path
        self.replication_path = replication_path
        self.chromatin_path = chromatin_path
        self.n_workers = n_workers

    def extract_data(self, bam_path: str) -> Tuple[pl.LazyFrame, pl.DataFrame]:
        """
        Extracts, filters, and summarizes data for one BAM file.
        """
        if self.verbose:
            print(f"[BamProcessor:INFO] Extracting data from {bam_path}", flush=True)

        chr_processor = ChromosomeProcessor(config=self.cfg,
                                            verbose=self.verbose,
                                            filters=self.filters,
                                            coverage_path=self.coverage_path,
                                            replication_path=self.replication_path,
                                            chromatin_path=self.chromatin_path,
                                            n_workers=self.n_workers)

        bed_path = self.cfg.get("bed_file")
        mutation_types = self.cfg.get("mutation_types")
        
        if not mutation_types:
            raise ValueError("cfg['mutation_types'] is missing or empty")
        
        lf, genome_stats = chr_processor.process(
            bam_path=bam_path,
            reference_path=self.reference_path,
            mutation_types=mutation_types,
            bed_path=bed_path,
        )
        
        if lf is None:
            raise RuntimeError("No data extracted")
                
        # Attach ids lazily
        patient_id = get_patient_id(bam_path)
        sample_id = get_sample_id(bam_path)
        lf = lf.with_columns([
            pl.lit(patient_id).alias("patient_id"),
            pl.lit(sample_id).alias("sample_id"),
        ])
      
        # Collect stats
        info = self.summarize_info(genome_stats.total)
        
        if self.verbose:
            print(f"[BamProcessor:INFO] Completed BAM: {bam_path}", flush=True)

        return lf, info

    @staticmethod
    def summarize_info(stats: ChromStats) -> pl.DataFrame:
        """Compute high-level summary stats."""
        total_mismatches = stats.n_mismatches
        total_matches = stats.n_matches
        total_coverage = stats.n_total

        summary = pl.DataFrame({
            "total_mismatches": [total_mismatches],
            "total_matches": [total_matches],
            "total_coverage": [total_coverage],
        })

        beta = summary["total_matches"][0] / max(summary["total_coverage"][0] - summary["total_mismatches"][0], 1e-8)
        summary = summary.with_columns(pl.Series("beta", [beta]))
        
        return summary