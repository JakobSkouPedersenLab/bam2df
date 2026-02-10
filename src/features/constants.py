import re
import polars as pl

OBS_SNV = "S"
OBS_INSERTION = "I"
OBS_DELETION = "D"
OBS_MATCH = "M"

_MD_RE = re.compile(r"(\d+)|([A-Z])|\^([A-Z]+)")
_MD_TOKEN_RE = re.compile(r"(\d+)|\^([A-Z]+)|([A-Z])")
_VALID_CHROMS = [f"chr{i}" for i in range(1, 23)]

_PL_SCHEMA = {
    "qname": pl.UInt64,
    "chr": pl.Utf8,
    "genomic_pos": pl.Int32,
    "obs_type": pl.Categorical,
    "obs_base": pl.Utf8,
    "obs_ref": pl.Utf8,
    "obs_alt_len": pl.Int8,
    "obs_ref_len": pl.Int8,
    "len_delta": pl.Int8,
    "strand": pl.Utf8,
    "first_in_pair": pl.Int8,
    "read_index": pl.Int16,
    "fragment_len": pl.Int16,
    "seq_length": pl.Int16,
    "trinucleotide_ctx": pl.Utf8,
    "start_ctx": pl.Utf8,
    "end_ctx": pl.Utf8,
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
_REPLICATION_TIMING_SCHEMA = {
    "replication_timing": pl.Float32,
}
_CHROMATIN_SCHEMA = {
    "atac_open": pl.Int8,
    "atac_dist_to_open": pl.UInt16,
    "dnase_open": pl.Int8,
    "dnase_dist_to_open": pl.UInt16,
}