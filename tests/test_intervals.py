from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from src.features.read import _pos_in_blacklist_chr
from src.utility.io import load_bed_intervals, merge_intervals, prepare_blacklist_arrays


def test_merge_intervals_merges_overlapping_nested_and_touching():
    starts, ends = merge_intervals([50, 0, 120, 200, 210], [60, 100, 130, 210, 220])
    # [0,100) swallows [50,60); [200,210) touches [210,220); [120,130) stands alone
    assert starts.tolist() == [0, 120, 200]
    assert ends.tolist() == [100, 130, 220]


def test_merge_intervals_handles_empty_input():
    starts, ends = merge_intervals([], [])
    assert starts.size == 0 and ends.size == 0


def test_blacklist_lookup_finds_position_covered_by_earlier_longer_interval():
    df = pl.DataFrame({"chr": ["chr1", "chr1"], "start": [0, 50], "end": [100, 60]})
    arrays = prepare_blacklist_arrays({"chr1": df})

    # 70 lies in [0,100) but after [50,60); before merging, the lookup checked only [50,60)
    assert _pos_in_blacklist_chr(arrays, "chr1", 70)
    assert _pos_in_blacklist_chr(arrays, "chr1", 99)
    assert not _pos_in_blacklist_chr(arrays, "chr1", 100)
    assert isinstance(arrays["chr1"][0], np.ndarray)


def test_bed_intervals_are_merged_so_overlaps_are_not_extracted_twice(tmp_path: Path):
    bed = tmp_path / "regions.bed"
    bed.write_text(
        "chr1\t500\t600\n"
        "chr1\t100\t200\n"
        "chr1\t150\t300\n"
        "chr1\t300\t350\n"
        "chr2\t10\t20\n"
    )
    assert load_bed_intervals(str(bed)) == {
        "chr1": [(100, 350), (500, 600)],
        "chr2": [(10, 20)],
    }
