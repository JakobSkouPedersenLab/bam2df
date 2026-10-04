from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from src.utility.variant_blacklist import (
    count_snv_reference_mismatches,
    load_variant_blacklist,
    parse_variant_blacklist,
)


def _calls(rows: list[tuple[str, str, str, str]], header=("#CHR", "START", "REF", "ALT")) -> pl.DataFrame:
    return pl.DataFrame([list(r) for r in rows], schema=list(header), orient="row")


def test_snvs_and_anchored_indels_become_extractor_keys():
    df = _calls([
        ("chr1", "99", "A", "G"),       # SNV at 0-based 99
        ("chr1", "199", "C", "CTT"),    # insertion of TT after anchor C
        ("chr2", "299", "GAA", "G"),    # deletion of AA after anchor G
    ])
    snv, indel, counts = parse_variant_blacklist(df, "test")

    assert snv == {"chr1": {(99, "A", "G")}}
    # end0 = start0 + len(ref), as extractors.py builds it from reads
    assert indel == {"chr1": {(199, 200, "C", "CTT")}, "chr2": {(299, 302, "GAA", "G")}}
    assert counts["snv"] == 1 and counts["indel"] == 2


def test_multiallelic_alt_is_split_and_case_and_chrom_are_normalized():
    df = _calls([("1", "10", "a", "c,t")])
    snv, _, counts = parse_variant_blacklist(df, "test")
    assert snv == {"chr1": {(10, "A", "C"), (10, "A", "T")}}
    assert counts["snv"] == 2


def test_alleles_bam2df_never_emits_are_counted_not_keyed():
    df = _calls([
        ("chr1", "10", "AC", "GT"),     # multi-base substitution
        ("chr1", "20", "AT", "GCC"),    # complex indel, no shared anchor
        ("chr1", "30", "A", "*"),       # spanning deletion
        ("chr1", "40", "A", "G"),
    ])
    snv, indel, counts = parse_variant_blacklist(df, "test")
    assert snv == {"chr1": {(40, "A", "G")}}
    assert indel == {}
    assert counts["skipped_multi_base_substitution"] == 1
    assert counts["skipped_complex_indel"] == 1
    assert counts["skipped_non_sequence_allele"] == 1


def test_end_column_is_ignored():
    df = _calls([("chr1", "99", "A", "G", "100")], header=("#CHR", "START", "REF", "ALT", "END"))
    snv, _, _ = parse_variant_blacklist(df, "test")
    assert snv == {"chr1": {(99, "A", "G")}}


@pytest.mark.parametrize(
    "rows, message",
    [
        ([("chr1", "x", "A", "G")], "not an integer"),
        ([("chr1", "-1", "A", "G")], "negative"),
        ([("chr1", "5", "A", "A")], "ALT equals REF"),
        ([("chr1", "5", "A", "R")], "not an ACGTN"),
        ([("chr1", "5", "", "G")], "not an ACGTN"),
        ([("chr1", "5", "AC", "GT")], "no usable"),
    ],
)
def test_invalid_rows_fail(rows, message):
    with pytest.raises(ValueError, match=message):
        parse_variant_blacklist(_calls(rows), "test")


def test_missing_required_column_fails():
    df = pl.DataFrame({"#CHR": ["chr1"], "POS": ["5"], "REF": ["A"], "ALT": ["G"]})
    with pytest.raises(ValueError, match="missing columns"):
        parse_variant_blacklist(df, "test")


def test_load_reads_tsv_and_fails_on_missing_file(tmp_path: Path):
    path = tmp_path / "P1_snpcalls.tsv"
    path.write_text("#CHR\tSTART\tEND\tREF\tALT\nchr3\t0007\t8\tT\tC\n")
    snv, _, _ = load_variant_blacklist(path)
    assert snv == {"chr3": {(7, "T", "C")}}

    with pytest.raises(FileNotFoundError):
        load_variant_blacklist(tmp_path / "absent.tsv")


def test_reference_check_counts_mismatches_deterministically():
    genome = {"chr1": "ACGTACGT"}
    snv = {"chr1": {(0, "A", "G"), (1, "C", "T"), (2, "T", "A")}}  # third REF is wrong: genome has G
    fetch = lambda chrom, pos0: genome[chrom][pos0]

    assert count_snv_reference_mismatches(snv, fetch, max_checks=10) == (1, 3)
    # Sorted order: the first two checked are positions 0 and 1, both correct.
    assert count_snv_reference_mismatches(snv, fetch, max_checks=2) == (0, 2)


def test_reference_check_detects_one_based_shift():
    genome = {"chr1": "ACGTTGCAACGGTTCA"}
    # Calls written 1-based: START is one past the true 0-based position.
    snv = {"chr1": {(pos0 + 1, base, "N") for pos0, base in enumerate(genome["chr1"][:-1])}}
    n_mismatched, n_checked = count_snv_reference_mismatches(snv, lambda c, p: genome[c][p], max_checks=100)
    assert n_mismatched / n_checked > 0.5
