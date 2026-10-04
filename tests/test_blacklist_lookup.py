from __future__ import annotations

import pickle

from src.features.read import _is_indel_blacklisted, _is_snv_blacklisted
from src.utility.data import FilterContext


def _context() -> FilterContext:
    return FilterContext(
        snv_blacklist_by_chr={"chr1": {(99, "A", "G")}, "chr2": {(5, "C", "T")}},
        indel_blacklist_by_chr={"chr1": {(199, 200, "C", "CTT")}},
    )


def test_snv_lookup_requires_exact_allele():
    ctx = _context()
    assert _is_snv_blacklisted(ctx, "chr1", 99, "A", "G")
    # A different base at a germline site is an error, not the germline allele: keep it.
    assert not _is_snv_blacklisted(ctx, "chr1", 99, "A", "T")
    assert not _is_snv_blacklisted(ctx, "chr1", 100, "A", "G")
    assert not _is_snv_blacklisted(ctx, "chr3", 99, "A", "G")
    assert not _is_snv_blacklisted(None, "chr1", 99, "A", "G")


def test_indel_lookup_requires_exact_allele():
    ctx = _context()
    assert _is_indel_blacklisted(ctx, "chr1", 199, 200, "C", "CTT")
    assert not _is_indel_blacklisted(ctx, "chr1", 199, 200, "C", "CT")


def test_for_chromosome_keeps_only_that_chromosome_and_pickles():
    ctx = _context()
    chr1 = pickle.loads(pickle.dumps(ctx.for_chromosome("chr1")))
    assert chr1.snv_blacklist_by_chr == {"chr1": {(99, "A", "G")}}
    assert chr1.indel_blacklist_by_chr == {"chr1": {(199, 200, "C", "CTT")}}
    assert _is_snv_blacklisted(chr1, "chr1", 99, "A", "G")

    chr3 = ctx.for_chromosome("chr3")
    assert chr3.snv_blacklist_by_chr == {} and chr3.indel_blacklist_by_chr == {}
    # The original context is unchanged.
    assert set(ctx.snv_blacklist_by_chr) == {"chr1", "chr2"}
