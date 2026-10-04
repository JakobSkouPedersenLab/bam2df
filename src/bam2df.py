#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import polars as pl
import pyBigWig
import pysam
import pickle
import pyarrow.parquet as pq

from scipy.sparse import csr_matrix, load_npz

from src.features.labeling import add_tumor_mutation_labels
from src.utility.data import FilterContext
from src.utility.sample import get_sample_id
from src.features.bam_processor import BamProcessor

from src.utility.config import validate_bam_config
from src.utility.variant_blacklist import count_snv_reference_mismatches, load_variant_blacklist
from src.utility.io import (load_bed, load_bed_intervals, load_table, load_lowpower_blacklist,
                            load_problematic_blacklist, prepare_blacklist_arrays, split_by_chr)
from src.utility.sample import get_patient_id

from src.features.constants import (
    _PL_SCHEMA,
    _REPLICATION_TIMING_SCHEMA,
    _MAPPABILITY_SCHEMA,
    _CHROMATIN_SCHEMA,
    _MICROSAT_SCHEMA,
    _REPMASK_SCHEMA,
    _VALID_CHROMS,
)

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
    parser.add_argument("--bed_intervals_path", type=str, default=None)
    parser.add_argument("--chromatin_path", type=str, default=None)
    parser.add_argument("--replication_path", type=str, default=None)
    parser.add_argument("--mappability_path", type=str, default=None)
    parser.add_argument("--microsatellite_path", type=str, default=None)
    parser.add_argument("--repeatmasker_path", type=str, default=None)
    
    parser.add_argument("--output_file", type=str, default=None)
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

    snv_by_chr, indel_by_chr = {}, {}
    if args.snp_blacklist_path:
        snv_by_chr, indel_by_chr = load_germline_calls(args.snp_blacklist_path, args.reference_path)

    patient_bl = {}
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
        snv_blacklist_by_chr=snv_by_chr,
        indel_blacklist_by_chr=indel_by_chr,
    )

# Up to this many germline SNVs are checked against the reference; above this mismatch fraction
# the calls file is assumed to use a different coordinate convention. Correct 0-based calls
# mismatch only where the caller and the FASTA disagree (rare); an off-by-one shift mismatches ~3/4.
_REF_CHECK_MAX_SNVS = 10_000
_REF_CHECK_MAX_MISMATCH_FRACTION = 0.05

def load_germline_calls(path: str, reference_path: str):
    """
    Load the patient's germline calls as exact-allele SNV and indel blacklists, restricted to the
    chromosomes bam2df processes, and verify their coordinates against the reference.
    """
    snv_by_chr, indel_by_chr, counts = load_variant_blacklist(path)
    snv_by_chr = {c: v for c, v in snv_by_chr.items() if c in _VALID_CHROMS}
    indel_by_chr = {c: v for c, v in indel_by_chr.items() if c in _VALID_CHROMS}
    print(f"[INFO] SNP blacklist {path}: {counts}; "
          f"kept on processed chromosomes: snv={sum(map(len, snv_by_chr.values())):,} "
          f"indel={sum(map(len, indel_by_chr.values())):,}", flush=True)

    with pysam.FastaFile(reference_path) as fasta:
        n_mismatched, n_checked = count_snv_reference_mismatches(
            snv_by_chr,
            fetch_ref_base=lambda chrom, pos0: fasta.fetch(chrom, pos0, pos0 + 1),
            max_checks=_REF_CHECK_MAX_SNVS,
        )
    if n_checked and n_mismatched / n_checked > _REF_CHECK_MAX_MISMATCH_FRACTION:
        raise ValueError(
            f"SNP blacklist {path}: REF disagrees with {reference_path} at {n_mismatched:,} of "
            f"{n_checked:,} checked SNVs. START must be 0-based on the same assembly as the reference."
        )
    return snv_by_chr, indel_by_chr

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

def order_columns_by_schema(lf: pl.LazyFrame) -> pl.LazyFrame:
    schema_order = [
        *_PL_SCHEMA.keys(),
        *_REPLICATION_TIMING_SCHEMA.keys(),
        *_MAPPABILITY_SCHEMA.keys(),
        *_CHROMATIN_SCHEMA.keys(),
        *_MICROSAT_SCHEMA.keys(),
        *_REPMASK_SCHEMA.keys(),
        "patient_id",
        "sample_id",
        "tumor_label",
        "somatic_label",
    ]
    available = lf.collect_schema().names()
    ordered = [c for c in schema_order if c in available]
    remaining = [c for c in available if c not in ordered]
    return lf.select(ordered + remaining)

def write_single_parquet_from_chunks(
    chunk_paths: list[Path],
    output_path: Path,
    patient_id: str,
    sample_id: str,
    mutation_file,
    use_chromatin: bool,
) -> None:
    if output_path.exists():
        output_path.unlink()

    writer = None

    try:
        for i, chunk_path in enumerate(chunk_paths):
            lf = pl.scan_parquet(chunk_path)

            if use_chromatin:
                chrom_path = chunk_path.with_suffix(".chrom.parquet")
                if chrom_path.exists():
                    lf_chrom = pl.scan_parquet(chrom_path)
                    lf = pl.concat([lf, lf_chrom], how="horizontal")

            lf = lf.with_columns([
                pl.lit(patient_id).cast(pl.Utf8).alias("patient_id"),
                pl.lit(sample_id).cast(pl.Utf8).alias("sample_id"),
            ])

            lf = add_tumor_mutation_labels(lf, mutation_file)
            lf = order_columns_by_schema(lf)

            try:
                df = lf.collect(engine="streaming")
            except TypeError:
                df = lf.collect(streaming=True)

            table = df.to_arrow()

            if writer is None:
                writer_schema = table.schema
                writer = pq.ParquetWriter(
                    output_path,
                    writer_schema,
                    compression="zstd",
                )
            else:
                table = table.select(writer_schema.names)

            writer.write_table(table)

            del lf, df, table

    finally:
        if writer is not None:
            writer.close()

def main():
    args = parse_args()
    
    # Load config
    with open(args.config_path) as f:
        cfg = json.load(f)
    bam_config = cfg['bam']
    validate_bam_config(bam_config)

    output_file = Path(args.output_file).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

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

    # Optional replication timing
    replication_path = (
        str(Path(args.replication_path).resolve())
        if args.replication_path and Path(args.replication_path).exists()
        else None
    )
    
    # Optional mappability
    mappability_path = (
        str(Path(args.mappability_path).resolve())
        if args.mappability_path and Path(args.mappability_path).exists()
        else None
    )

    # Optional microsatellite
    microsatellite_path = (
        str(Path(args.microsatellite_path).resolve())
        if args.microsatellite_path and Path(args.microsatellite_path).exists()
        else None
    )

    # Optional RepeatMasker
    repeatmasker_path = (
        str(Path(args.repeatmasker_path).resolve())
        if args.repeatmasker_path and Path(args.repeatmasker_path).exists()
        else None
    )

    # Optional filters
    patient_id = get_patient_id(args.bam_path)
    sample_id = get_sample_id(args.bam_path)
    filters = load_blacklists(args, patient_id)
    
    # Optional BED filters
    bed_intervals_path = (
        str(Path(args.bed_intervals_path).resolve())
        if args.bed_intervals_path and Path(args.bed_intervals_path).exists()
        else None
    )
    bed_intervals = load_bed_intervals(bed_intervals_path) if bed_intervals_path else None

    processor = BamProcessor(cfg=bam_config,
                             reference_path=args.reference_path,
                             verbose=args.verbose,
                             filters=filters,
                             replication_path=replication_path,
                             chromatin_path=chromatin_path,
                             mappability_path=mappability_path,
                             microsatellite_path=microsatellite_path,
                             repeatmasker_path=repeatmasker_path,
                             bed_intervals=bed_intervals,
                             n_workers=n_workers)
    chunk_paths, df_info = processor.extract_data(args.bam_path)
    
    write_single_parquet_from_chunks(
        chunk_paths=chunk_paths,
        output_path=output_file,
        patient_id=patient_id,
        sample_id=sample_id,
        mutation_file=mutation_file,
        use_chromatin=chromatin_path is not None,
    )

    if args.verbose:
        print(f"[INFO] Completed extraction -> {output_file}", flush=True)
        
    # Metadata CSV beside parquet file
    meta_path = write_metadata_per_file(df_info, output_file)

    if args.verbose:
        print(f"[INFO] Metadata appended -> {meta_path}", flush=True)

if __name__ == "__main__":
    main()
