from typing import Optional, Tuple
import polars as pl

from src.features.feature_extractor import ReadFeatureExtractor
from src.utility.sample import get_patient_id, get_sample_id
from src.utility.data import ChromStats, FilterContext, GenomeStats

import pyBigWig

class BamProcessor:
    """
    High-level orchestrator for processing one BAM file into model-ready data.
    Wraps ReadFeatureExtractor and applies post-extraction filtering, sampling,
    and feature normalization.
    """
    def __init__(self, cfg,
                 reference_path,
                 verbose: bool = False,
                 filters: Optional[FilterContext] = None,
                 coverage_bw: Optional["pyBigWig.bigWigFile"] = None):
        
        self.cfg = cfg
        self.reference_path = reference_path
        self.verbose = verbose
        self.filters = filters
        self.coverage_bw = coverage_bw

    def extract_data(self, bam_path: str) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Extracts, filters, and summarizes data for one BAM file.
        """
        if self.verbose:
            print(f"[BamProcessor:INFO] Extracting data from {bam_path}", flush=True)

        extractor = ReadFeatureExtractor(
            bam_path,
            config=self.cfg,
            reference_path=self.reference_path,
            verbose=self.verbose,
            filters=self.filters,
            coverage_bw=self.coverage_bw,
        )

        patient_id = get_patient_id(bam_path)
        sample_id = get_sample_id(bam_path)

        bed_path = self.cfg.get("bed_file")
        extract_type = self.cfg.get("extract_type")

        # Extract mismatch data (if requested)
        mismatches_df = pl.DataFrame()
        mismatch_stats = None

        if extract_type == "mismatch":
            mismatches_df, mismatch_stats = self._extract_df(
                extractor,
                bam_path,
                "mismatch",
                patient_id,
                sample_id,
                bed_path,
            )

        # Extract match data (if requested)
        matches_df = pl.DataFrame()
        match_stats = None

        if extract_type == "match":
            factor = self.cfg.get("factor", 1.0)
            n_matches = int(len(mismatches_df) * factor)
            matches_df, match_stats = self._extract_df(
                extractor,
                bam_path,
                "match",
                patient_id,
                sample_id,
                bed_path,
                n_match_sample=n_matches,
            )

        # Extract indel data (if requested)
        indels_df = pl.DataFrame()
        indel_stats = None

        if extract_type == "indel":
            indels_df, indel_stats = self._extract_df(
                extractor,
                bam_path,
                "indel",
                patient_id,
                sample_id,
                bed_path,
            )

        # Combine data frames depending on extract type
        if extract_type == "mismatch":
            data = mismatches_df
        elif extract_type == "match":
            data = matches_df
        else:
            data = indels_df

        # Combine stats
        total_stats = ChromStats()

        for stats in (mismatch_stats, match_stats, indel_stats):
            if stats is not None:
                total_stats += stats.total

        info = self.summarize_info(total_stats)

        if self.verbose:
            print(f"[BamProcessor:INFO] Completed BAM: {bam_path}", flush=True)

        return data, info

    def _extract_df(self,
                    extractor,
                    bam_path: str,
                    kind: str,
                    patient_id: str,
                    sample_id: str,
                    bed_path: str,
                    **kwargs) -> Tuple[pl.DataFrame, Optional[ChromStats]]:
        """
        Extracts one type (mismatch/match/indel), attaches patient/sample IDs,
        and returns (df, stats). Returns (empty_df, None) if extraction yields nothing.
        """
        df, stats = extractor.extract(bam_path, kind, bed_path=bed_path, **kwargs)

        if df.is_empty():
            if self.verbose:
                print(f"[BamProcessor:WARN] No {kind} data extracted.", flush=True)
            return pl.DataFrame(), ChromStats()

        n_rows = df.height
        id_df = pl.DataFrame({
            "patient_id": [patient_id] * n_rows,
            "sample_id":  [sample_id] * n_rows,
        })

        df = pl.concat([df, id_df], how="horizontal")
        return df, stats

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