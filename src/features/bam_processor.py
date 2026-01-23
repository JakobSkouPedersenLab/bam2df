from typing import Optional, Tuple
import polars as pl

from src.features.chr_processor import ChromosomeProcessor
from src.utility.sample import get_patient_id, get_sample_id
from src.utility.data import ChromStats, ChromatinResources, FilterContext, GenomeStats

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
                 chromatin: ChromatinResources = None,
                 n_workers: int = 1):
        
        self.cfg = cfg
        self.reference_path = reference_path
        self.verbose = verbose
        self.filters = filters
        self.coverage_path = coverage_path
        self.replication_path = replication_path
        self.chromatin = chromatin
        self.n_workers = n_workers

    def extract_data(self, bam_path: str) -> Tuple[pl.DataFrame, pl.DataFrame]:
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
                                            chromatin=self.chromatin,
                                            n_workers=self.n_workers)

        patient_id = get_patient_id(bam_path)
        sample_id = get_sample_id(bam_path)

        bed_path = self.cfg.get("bed_file")
        mutation_types = self.cfg.get("mutation_types")
        
        # Exctract data
        df, stats = self._extract_df(
            chr_processor=chr_processor,
            bam_path=bam_path,
            mutation_types=mutation_types,
            patient_id=patient_id,
            sample_id=sample_id,
            bed_path=bed_path,
        )
        
        # Collect stats
        total_stats = ChromStats()
        total_stats += stats.total
        info = self.summarize_info(total_stats)
        
        if self.verbose:
            print(f"[BamProcessor:INFO] Completed BAM: {bam_path}", flush=True)

        return df, info

    def _extract_df(self,
                    chr_processor: ChromosomeProcessor,
                    bam_path: str,
                    mutation_types: list[str],
                    patient_id: str,
                    sample_id: str,
                    bed_path: str,
                    **kwargs) -> Tuple[pl.DataFrame, Optional[ChromStats]]:
        """
        Extracts one type (mismatch/match/indel), attaches patient/sample IDs,
        and returns (df, stats). Returns (empty_df, None) if extraction yields nothing.
        """
        df, genome_stats = chr_processor.process(
            bam_path=bam_path,
            reference_path=self.reference_path,
            mutation_types=mutation_types,
            bed_path=bed_path,
        )
        
        if df.is_empty():
            if self.verbose:
                print(f"[BamProcessor:WARN] No {mutation_types} data extracted.", flush=True)
            return pl.DataFrame(), ChromStats()

        n_rows = df.height
        id_df = pl.DataFrame({
            "patient_id": [patient_id] * n_rows,
            "sample_id":  [sample_id] * n_rows,
        })

        df = pl.concat([df, id_df], how="horizontal")
        return df, genome_stats

    @staticmethod
    def combine_mismatches_matches(pos: pl.DataFrame,
                                   neg: pl.DataFrame | None) -> pl.DataFrame:
        """Concatenate positive and negative sets (if available)."""
        if neg is None or neg.is_empty():
            return pos
        return pl.concat([pos, neg])

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