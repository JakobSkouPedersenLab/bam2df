import polars as pl
from collections import defaultdict

def add_tumor_mutation_labels(data: pl.DataFrame, mutation_file) -> pl.DataFrame:
    patient_id = data["patient_id"][0]
    
    if mutation_file is None or len(mutation_file) == 0:
        return data.with_columns(pl.lit(0).cast(pl.Int8).alias("tumor_mut_present"))

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
    labels = [
        int((pos, ref, obs) in tumor_dict[chr_])
        for chr_, pos, ref, obs in zip(
            data["chr"], data["genomic_pos"], data["ref"], data["obs"]
        )
    ]

    return data.with_columns(
        pl.Series("tumor_mut_present", labels).cast(pl.Int8)
    )