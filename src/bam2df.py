#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import polars as pl
import pyBigWig
import pickle

from scipy.sparse import csr_matrix, load_npz

from src.features.labeling import add_tumor_mutation_labels
from src.utility.data import FilterContext
from src.utility.sample import get_sample_id
from src.features.bam_processor import BamProcessor

from src.utility.io import (load_bed, load_table, load_snp_blacklist, load_lowpower_blacklist,
                            load_problematic_blacklist, prepare_blacklist_arrays, split_by_chr)
from src.utility.sample import get_patient_id

n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

def parse_args():
    parser = argparse.ArgumentParser(description="Process a single BAM file into model-ready data.")

    parser.add_argument("bam_path", type=str, help="Path to BAM file")
    parser.add_argument("reference_path", type=str, help="Reference FASTA")
    parser.add_argument("config_path", type=str, help="Config JSON")

    parser.add_argument("--mutation_path", type=str, default=None)
    parser.add_argument("--genomic_blacklist_file", type=str, default=None)
    parser.add_argument("--donor_blacklist_file", type=str, default=None)
    parser.add_argument("--snp_blacklist_path", type=str, default=None)
    parser.add_argument("--lowpower_blacklist_path", type=str, default=None)
    parser.add_argument("--problematic_blacklist_path", type=str, default=None)
    parser.add_argument("--coverage_path", type=str, default=None)
    parser.add_argument("--chromatin_path", type=str, default=None)
    parser.add_argument("--replication_path", type=str, default=None)
    
    parser.add_argument("--output_file", type=str, default="./data/output.parquet")
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()

def load_blacklists(args, patient_id: str):
    """
    Load all blacklist resources and return a FilterContext ready for BamProcessor.
    """
    genomic_df = load_bed(args.genomic_blacklist_file, "genomic blacklist") \
                 if args.genomic_blacklist_file else None
    donor_df = load_table(args.donor_blacklist_file, "donor blacklist") \
               if args.donor_blacklist_file else None
    genomic_by_chr = prepare_blacklist_arrays(split_by_chr(genomic_df)) \
                     if genomic_df is not None else None
    donor_by_chr = prepare_blacklist_arrays(split_by_chr(donor_df)) \
                   if donor_df is not None else None

    patient_bl = {}
    if args.snp_blacklist_path:
        snp_df = load_snp_blacklist(args.snp_blacklist_path, patient_id)
        patient_bl["snp_blacklist"] = (
            prepare_blacklist_arrays(split_by_chr(snp_df))
            if snp_df is not None else None
        )
    if args.lowpower_blacklist_path:
        lp_df = load_lowpower_blacklist(args.lowpower_blacklist_path, patient_id)
        patient_bl["lowpower_blacklist"] = (
            prepare_blacklist_arrays(split_by_chr(lp_df))
            if lp_df is not None else None
        )
    if args.problematic_blacklist_path:
        prob_df = load_problematic_blacklist(args.problematic_blacklist_path, patient_id)
        patient_bl["problematic_blacklist"] = (
            prepare_blacklist_arrays(split_by_chr(prob_df))
            if prob_df is not None else None
        )

    patient_blacklists_by_chr = {
        k: v for k, v in patient_bl.items() if v is not None
    } or None

    return FilterContext(
        genomic_blacklist_by_chr=genomic_by_chr,
        donor_blacklist_by_chr=donor_by_chr,
        patient_blacklists_by_chr=patient_blacklists_by_chr,
    )

def write_metadata_per_file(info_df: pl.DataFrame, output_file: Path):
    """
    Write metadata for a single BAM output.
    Metadata file is written beside the parquet file:
        <sample_id>_metadata.csv
    """
    out_dir = output_file.parent
    sample_id = output_file.stem
    meta_path = out_dir / f"{sample_id}_metadata.csv"
    
    sample_id = get_sample_id(str(output_file))
    
    # Add sample_id column
    info_df = info_df.with_columns(
        pl.lit(sample_id).alias("sample_id")
    )
    
    # Ensure sample_id is first column
    info_df = info_df.select(
        ["sample_id"] + [c for c in info_df.columns if c != "sample_id"]
    )

    info_df.write_csv(meta_path)
    return meta_path

def main():
    args = parse_args()
    
    # Load config
    with open(args.config_path) as f:
        cfg = json.load(f)
    bam_config = cfg['bam']

    # Optional mutations file
    mutation_file = None
    if args.mutation_path and args.mutation_path.lower() != "none":
        if os.path.exists(args.mutation_path):
            with open(args.mutation_path, "rb") as f:
                mutation_file = pickle.load(f)
        else:
            print(f"[INFO] No mutation file found at {args.mutation_path}, continuing without it.")
    else:
        print("[INFO] No mutation file provided, continuing without mutation index.")
        
    # Optional chromatin
    chromatin_path = (
        str(Path(args.chromatin_path).resolve())
        if args.chromatin_path and Path(args.chromatin_path).exists()
        else None
    )

    # Optional coverage
    coverage_path = (
        str(Path(args.coverage_path).resolve())
        if args.coverage_path and Path(args.coverage_path).exists()
        else None
    )

    # Optional replication timing
    replication_path = (
        str(Path(args.replication_path).resolve())
        if args.replication_path and Path(args.replication_path).exists()
        else None
    )

    # Optional filters
    patient_id = get_patient_id(args.bam_path)
    filters = load_blacklists(args, patient_id)

    processor = BamProcessor(cfg=bam_config,
                             reference_path=args.reference_path,
                             verbose=args.verbose,
                             filters=filters,
                             coverage_path=coverage_path,
                             replication_path=replication_path,
                             chromatin_path=chromatin_path,
                             n_workers=n_workers)
    lf_data, df_info = processor.extract_data(args.bam_path)
    
    # Add tumor mutation labels
    lf_data = add_tumor_mutation_labels(lf_data, mutation_file)

    # Write parquet file
    output_file = args.output_file
    lf_data.sink_parquet(output_file, compression="zstd")

    if args.verbose:
        print(f"[INFO] Completed extraction -> {output_file}", flush=True)
        
    # Metadata CSV beside parquet file
    meta_path = write_metadata_per_file(df_info, output_file)

    if args.verbose:
        print(f"[INFO] Metadata appended -> {meta_path}", flush=True)

if __name__ == "__main__":
    main()
