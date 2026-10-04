"""Patient germline variant calls as exact-allele blacklists.

An observation is dropped only when its chromosome, position, REF and ALT all match a call, so a
sequencing error at a germline SNP site (a third base) is still extracted.

Coordinates: the calls file's START is 0-based, the snpcaller convention, and REF/ALT are
VCF-style, with indels anchored on the preceding reference base. The keys built here are the
ones the extractors build from reads:

- SNV: ``(start0, ref, alt)``
- indel: ``(start0, end0, ref, alt)`` with ``end0 = start0 + len(ref)``

The file's END column, if present, is not used.

This module imports only Polars so it can be tested without pysam.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import polars as pl

SnvKey = tuple[int, str, str]
IndelKey = tuple[int, int, str, str]
SnvBlacklist = dict[str, set[SnvKey]]
IndelBlacklist = dict[str, set[IndelKey]]

_REQUIRED_COLUMNS = ("chr", "start", "ref", "alt")
_ALLELE_RE = re.compile(r"^[ACGTN]+$")
# Alleles that name no observable sequence: VCF spanning deletion and missing allele.
_NON_SEQUENCE_ALLELES = {"*", "."}


def _normalize_chrom(chrom: str) -> str:
    chrom = chrom.strip()
    return chrom if chrom.startswith("chr") else f"chr{chrom}"


def parse_variant_blacklist(df: pl.DataFrame, source: str) -> tuple[SnvBlacklist, IndelBlacklist, dict[str, int]]:
    """Build exact-allele SNV and indel blacklists from a table of variant calls.

    Args:
        df: Calls with columns CHR, START, REF and ALT (case-insensitive, optional leading ``#``),
            read as strings. ALT may list several alleles separated by commas.
        source: Where the table came from, for error messages.

    Returns:
        ``(snv_by_chr, indel_by_chr, counts)``. ``counts`` records how many alleles became SNV and
        indel keys and how many were skipped, by reason.

    Raises:
        ValueError: A required column is missing, a START is not a non-negative integer, an allele
            contains characters other than ACGTN, REF equals ALT, or no usable allele remains.
    """
    df = df.rename({c: c.strip().lower().lstrip("#") for c in df.columns})
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"SNP blacklist {source} is missing columns {missing}; found {df.columns}")

    snv_by_chr: SnvBlacklist = {}
    indel_by_chr: IndelBlacklist = {}
    counts = {
        "snv": 0,
        "indel": 0,
        # bam2df emits single-base SNVs only, so a multi-base substitution can never match.
        "skipped_multi_base_substitution": 0,
        # bam2df emits anchored pure insertions/deletions only, so these can never match.
        "skipped_complex_indel": 0,
        "skipped_non_sequence_allele": 0,
    }

    for line_idx, (chrom, start, ref, alts) in enumerate(df.select(_REQUIRED_COLUMNS).iter_rows()):
        where = f"{source} data row {line_idx + 1}"
        try:
            start0 = int(start)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{where}: START {start!r} is not an integer") from e
        if start0 < 0:
            raise ValueError(f"{where}: START {start0} is negative")

        ref = (ref or "").strip().upper()
        if not _ALLELE_RE.match(ref):
            raise ValueError(f"{where}: REF {ref!r} is not an ACGTN sequence")
        chrom = _normalize_chrom(chrom or "")

        for alt in (alts or "").split(","):
            alt = alt.strip().upper()
            if alt in _NON_SEQUENCE_ALLELES:
                counts["skipped_non_sequence_allele"] += 1
                continue
            if not _ALLELE_RE.match(alt):
                raise ValueError(f"{where}: ALT {alt!r} is not an ACGTN sequence")
            if alt == ref:
                raise ValueError(f"{where}: ALT equals REF ({ref})")

            if len(ref) == len(alt):
                if len(ref) == 1:
                    snv_by_chr.setdefault(chrom, set()).add((start0, ref, alt))
                    counts["snv"] += 1
                else:
                    counts["skipped_multi_base_substitution"] += 1
            elif ref.startswith(alt) or alt.startswith(ref):
                # Shared prefix is the anchor base: alt = ref + inserted, or ref = alt + deleted.
                indel_by_chr.setdefault(chrom, set()).add((start0, start0 + len(ref), ref, alt))
                counts["indel"] += 1
            else:
                counts["skipped_complex_indel"] += 1

    if counts["snv"] == 0 and counts["indel"] == 0:
        raise ValueError(f"SNP blacklist {source} contains no usable SNV or indel alleles: {counts}")
    return snv_by_chr, indel_by_chr, counts


def load_variant_blacklist(path: str | Path) -> tuple[SnvBlacklist, IndelBlacklist, dict[str, int]]:
    """Read a tab-separated variant-calls file and build exact-allele blacklists.

    See ``parse_variant_blacklist`` for the expected columns and return value.

    Raises:
        FileNotFoundError: ``path`` is not a file.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"SNP blacklist not found: {p}")
    # Read every column as a string so START parsing and allele checks report the offending row.
    df = pl.read_csv(p, separator="\t", infer_schema_length=0)
    return parse_variant_blacklist(df, source=str(p))


def count_snv_reference_mismatches(
    snv_by_chr: SnvBlacklist,
    fetch_ref_base: Callable[[str, int], str],
    max_checks: int,
) -> tuple[int, int]:
    """Check SNV REF alleles against the reference genome, to catch a coordinate-convention error.

    A file whose START is 1-based instead of 0-based shifts every call by one base, and then
    roughly three in four REF alleles disagree with the genome.

    Args:
        snv_by_chr: SNV keys by chromosome.
        fetch_ref_base: Returns the reference base at a 0-based position on a chromosome.
        max_checks: Upper bound on the number of SNVs checked. The first ones in sorted
            (chromosome, position) order are used, so the check is deterministic.

    Returns:
        ``(n_mismatched, n_checked)``.
    """
    n_checked = 0
    n_mismatched = 0
    for chrom in sorted(snv_by_chr):
        for start0, ref, _alt in sorted(snv_by_chr[chrom]):
            if n_checked >= max_checks:
                return n_mismatched, n_checked
            if fetch_ref_base(chrom, start0).upper() != ref:
                n_mismatched += 1
            n_checked += 1
    return n_mismatched, n_checked
