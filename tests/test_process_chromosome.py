"""process_chromosome on a small BAM written here. Needs pysam, so it runs on the server only."""

from __future__ import annotations

import random
from pathlib import Path

import polars as pl
import pytest

from tests.conftest import requires_pysam

pytestmark = requires_pysam

CHROM = "chr1"
CHROM_LEN = 2000
READ_LEN = 50

# One pair: read1 forward on [100, 150), read2 reverse on [120, 170); overlap [120, 150).
READ1_START = 100
READ2_START = 120
FRAGMENT_LEN = READ2_START + READ_LEN - READ1_START  # 70


def _reference() -> str:
    rng = random.Random(7)
    return "".join(rng.choice("ACGT") for _ in range(CHROM_LEN))


def _other_base(base: str) -> str:
    return "C" if base != "C" else "G"


def _read(header, ref: str, name: str, flag: int, start: int, mate_start: int, tlen: int,
          mismatch_at: int | None):
    """A 50M read copied from the reference, with an optional mismatch at reference position `mismatch_at`."""
    import pysam

    seq = list(ref[start:start + READ_LEN])
    if mismatch_at is None:
        md, nm = str(READ_LEN), 0
    else:
        i = mismatch_at - start
        md, nm = f"{i}{seq[i]}{READ_LEN - i - 1}", 1
        seq[i] = _other_base(seq[i])

    a = pysam.AlignedSegment(header)
    a.query_name = name
    a.flag = flag
    a.reference_name = CHROM
    a.reference_start = start
    a.mapping_quality = 60
    a.cigarstring = f"{READ_LEN}M"
    a.next_reference_name = CHROM
    a.next_reference_start = mate_start
    a.template_length = tlen
    a.query_sequence = "".join(seq)
    a.query_qualities = pysam.qualitystring_to_array("I" * READ_LEN)  # Q40
    a.set_tag("MD", md)
    a.set_tag("NM", nm)
    return a


def _write_inputs(tmp_path: Path, read1_mismatch: int, read2_mismatch: int) -> tuple[Path, Path, str]:
    """Write a FASTA and an indexed BAM holding one read pair. Returns (bam, fasta, reference)."""
    import pysam

    ref = _reference()
    fasta = tmp_path / "ref.fa"
    fasta.write_text(f">{CHROM}\n{ref}\n")
    pysam.faidx(str(fasta))

    header = pysam.AlignmentHeader.from_dict(
        {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": CHROM, "LN": CHROM_LEN}]}
    )
    bam = tmp_path / "reads.bam"
    with pysam.AlignmentFile(str(bam), "wb", header=header) as out:
        # 99 = paired, proper, mate reverse, read1; 147 = paired, proper, reverse, read2
        out.write(_read(header, ref, "pair1", 99, READ1_START, READ2_START, FRAGMENT_LEN, read1_mismatch))
        out.write(_read(header, ref, "pair1", 147, READ2_START, READ1_START, -FRAGMENT_LEN, read2_mismatch))
    pysam.index(str(bam))
    return bam, fasta, ref


def _config(**overrides) -> dict:
    cfg = {
        "context_flank": 5,
        "min_len": 0,
        "max_len": 299,
        "min_base_quality": 30,
        "min_mapq": 20,
        "pre_sample_seed": 0,
        "n_match_sample": 0,
        "mutation_types": ["mismatch"],
        "use_paired_reads": True,
        "chunk_size": 1000,
    }
    cfg.update(overrides)
    return cfg


def _run(bam: Path, fasta: Path, tmp_dir: Path, cfg: dict):
    import pysam

    from src.features.chr_processor import process_chromosome
    from src.utility.data import FilterContext

    tmp_dir.mkdir(exist_ok=True)
    with pysam.AlignmentFile(str(bam), "rb") as bam_file:
        return process_chromosome(
            chrom=CHROM, start=0, end=CHROM_LEN, bam_file=bam_file, fasta_path=str(fasta),
            mutation_types=cfg["mutation_types"], n_match_sample=cfg["n_match_sample"],
            tmp_dir=tmp_dir, config=cfg, verbose=False, filters=FilterContext(),
            replication_bw=None, mappability_bw=None, microsatellite_lookup=None,
            repeatmasker_lookup=None, patient_id="P1", sample_id="S1",
        )


def _rows(chunk_paths: list[Path]) -> pl.DataFrame:
    return pl.concat([pl.read_parquet(p) for p in chunk_paths])


def test_paired_mode_reports_the_driver_mismatch_in_the_overlap(tmp_path: Path):
    bam, fasta, ref = _write_inputs(tmp_path, read1_mismatch=105, read2_mismatch=130)
    chunk_paths, stats = _run(bam, fasta, tmp_path / "work", _config())

    rows = _rows(chunk_paths)
    assert rows["genomic_pos"].to_list() == [131]
    assert rows["obs_ref"].to_list() == [ref[130]]
    assert rows["fragment_len"].to_list() == [FRAGMENT_LEN]
    assert stats.n_reads == 1 and stats.n_mismatches == 1


def test_single_read_mode_extracts_each_read(tmp_path: Path):
    # This mode used to fail with NameError: highqual_overlap was never set on its path
    bam, fasta, _ = _write_inputs(tmp_path, read1_mismatch=105, read2_mismatch=130)
    chunk_paths, stats = _run(bam, fasta, tmp_path / "work", _config(use_paired_reads=False))

    rows = _rows(chunk_paths).sort("genomic_pos")
    assert rows["genomic_pos"].to_list() == [106, 131]
    # The reverse read has TLEN -70; the fragment length is its absolute value
    assert rows["fragment_len"].to_list() == [FRAGMENT_LEN, FRAGMENT_LEN]
    assert rows["strand"].to_list() == ["fwd", "rev"]
    assert stats.n_reads == 2 and stats.n_invalid_fragments == 0



def test_match_subsampling_is_reproducible(tmp_path: Path):
    bam, fasta, _ = _write_inputs(tmp_path, read1_mismatch=105, read2_mismatch=130)
    cfg = _config(mutation_types=["match"], n_match_sample=5, pre_sample_seed=3)

    first, _ = _run(bam, fasta, tmp_path / "run1", cfg)
    second, _ = _run(bam, fasta, tmp_path / "run2", cfg)

    first_pos = _rows(first)["genomic_pos"].to_list()
    assert len(first_pos) == 5
    assert first_pos == _rows(second)["genomic_pos"].to_list()
