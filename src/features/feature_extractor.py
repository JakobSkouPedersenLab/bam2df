# extractor.py
import os
import time
import psutil
import tempfile
import polars as pl
from typing import Optional
from contextlib import contextmanager
import pysam
import pyBigWig

from src.utility.data import ChromStats, FilterContext
from src.features.chr_processor import ChromosomeProcessor

class ReadFeatureExtractor:
    """
    Handles low-level read-level feature extraction for a single BAM file.
    Encapsulates the logic previously scattered across extract_read_level_features(),
    process_chromosomes(), and _process_chromosome().
    """
    def __init__(self,
                 bam_path: str,
                 config: dict,
                 reference_path: str,
                 verbose: bool = False,
                 filters: Optional[FilterContext] = None,
                 coverage_bw: Optional["pyBigWig.bigWigFile"] = None):
        
        self.bam_path = bam_path
        self.config = config
        self.reference_path = reference_path
        self.verbose = verbose
        self.filters = filters
        self.coverage_bw = coverage_bw
        
    def extract(
        self,
        bam_path: str,
        extract_type: str,
        bed_path: Optional[str] = None,
        n_match_sample: int = 0,
        samtools_bin: str = "samtools",
        n_proc: Optional[int] = None,
    ) -> tuple[pl.DataFrame, ChromStats]:
        """
        Extracts features from one BAM file and returns a Polars DataFrame.
        """
        if self.verbose:
            print(f"[ReadFeatureExtractor] {os.path.basename(bam_path)} ({extract_type})")

        start = time.perf_counter()
        process = psutil.Process()

        processor = ChromosomeProcessor(self.config, 
                                        verbose=self.verbose,
                                        filters=self.filters,
                                        coverage_bw=self.coverage_bw)
        df, genome_stats = processor.process(
            bam_path=bam_path,
            fasta_path=self.reference_path,
            extract_type=extract_type,
            bed_path=bed_path,
        )
        
        elapsed = time.perf_counter() - start
        mem_info = process.memory_info()
        ram_gb = mem_info.rss / (1024 ** 3)
        
        if self.verbose:
            print(f"[ReadFeatureExtractor:INFO] Done in {elapsed:.1f}s "
                  f"({elapsed/60:.2f} min), RAM: {ram_gb:.2f} GB", flush=True)
                
        # Handle empty result
        if df.is_empty():
            if self.verbose:
                print(f"[ReadFeatureExtractor:WARN] No features extracted from {bam_path}", flush=True)
            
        if self.verbose:
            print(f"[ReadFeatureExtractor:INFO] Loaded {df.height:,} rows from {bam_path}", flush=True)

        return df, genome_stats
        
    # Internal helpers
    def _create_tmp_parquet(self) -> str:
        tmp_file = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()
        return tmp_path
