import pickle
import polars as pl
from pathlib import Path
from collections import defaultdict

def add_tumor_mutation_labels(df: pl.DataFrame, mutation_file) -> pl.DataFrame:
    patient_id = df["patient_id"][0]
    
    if mutation_file is None or len(mutation_file) == 0:
        return df.with_columns(pl.lit(0).cast(pl.Int8).alias("tumor_mut_present"))

    # Build dict-of-sets by chromosome for faster lookup
    tumor_dict = defaultdict(set)
    for entry in mutation_file:
        # Validate
        if not isinstance(entry, (tuple, list)) or len(entry) != 4:
            print(f"[WARN] Bad mutation entry for {patient_id}: {entry}", flush=True)
            continue

        chr_, pos, ref, obs = entry
        tumor_dict[chr_].add((pos, ref, obs))

    # Create labels by membership testing
    labels = []
    for chr_, pos, ref, obs in zip(df["chr"], df["genomic_pos"], df["obs_ref"], df["obs_base"]):
        is_tumor = (pos, ref, obs) in tumor_dict[chr_]
        labels.append(int(is_tumor))

    return df.with_columns(
        pl.Series("tumor_mut_present", labels).cast(pl.Int8)
    )