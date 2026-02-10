import pickle
from typing import Optional, Sequence, Tuple, Union
import polars as pl
from pathlib import Path
from collections import defaultdict

def add_tumor_mutation_labels(
    df: Union[pl.DataFrame, pl.LazyFrame],
    mutation_file: Optional[Sequence[Tuple[str, int, str, str]]],
) -> pl.LazyFrame:
    """
    Add tumor_mut_present (Int8 0/1) by joining on (chr, genomic_pos, obs_ref, obs_base).
    Works on DataFrame or LazyFrame; returns LazyFrame.
    """
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df

    if not mutation_file:
        return lf.with_columns(pl.lit(0).cast(pl.Int8).alias("tumor_mut_present"))

    # ensure we have a list of 4-tuples
    rows = []
    for entry in mutation_file:
        if not isinstance(entry, (tuple, list)) or len(entry) != 4:
            continue
        chr_, pos, ref, obs = entry
        rows.append((chr_, pos, ref, obs))

    if not rows:
        return lf.with_columns(pl.lit(0).cast(pl.Int8).alias("tumor_mut_present"))

    mut_df = pl.DataFrame(
        rows,
        schema=["chr", "genomic_pos", "obs_ref", "obs_base"],
        orient="row",
    ).unique()

    mut_lf = mut_df.lazy().with_columns(
        pl.lit(1).cast(pl.Int8).alias("tumor_mut_present")
    )

    return (
        lf.join(mut_lf, on=["chr", "genomic_pos", "obs_ref", "obs_base"], how="left")
          .with_columns(pl.col("tumor_mut_present").fill_null(0).cast(pl.Int8))
    )