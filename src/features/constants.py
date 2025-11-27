import re
import polars as pl

_MD_RE = re.compile(r"(\d+)|([A-Z])|\^([A-Z]+)")
_VALID_CHROMS = [f"chr{i}" for i in range(1, 2)] #1, 23
_PL_SCHEMA = {
    "qname": pl.Utf8,
    "chr": pl.Utf8,
    "genomic_pos": pl.Int32,
    "obs": pl.Utf8,
    "ref": pl.Utf8,
    "strand": pl.Utf8,
    "first_in_pair": pl.Int8,
    "read_index": pl.Int16,
    "fragment_len": pl.Int16,
    "seq_length": pl.Int16,
    "ctx_minus1": pl.Utf8,
    "ctx_plus1": pl.Utf8,
    "trinucleotide_ctx": pl.Utf8,
    "start_context": pl.Utf8,
    "end_context": pl.Utf8,
    "reference_ctx": pl.Utf8,
    "local_gc": pl.Float32,
    "local_complexity_1": pl.Float32,
    "local_complexity_2": pl.Float32,
    "n_other_err": pl.Int8,
    "base_qual_site": pl.Int8,
    "dist_to_end": pl.Int16,
    "homopolymer_run_len": pl.Int16,
    "read_depth": pl.Int16,
}