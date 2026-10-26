from __future__ import annotations

from pathlib import Path

from src.utility import io
from src.utility.io import get_hpc_tmp_dir, load_microsatellites_for_chrom, load_repeatmasker_for_chrom


def test_annotation_lookups_exist_for_a_chromosome_without_rows(tmp_path: Path):
    # Without a lookup the chromosome's chunks would lack the annotation columns
    microsat = tmp_path / "simpleRepeat.txt"
    microsat.write_text("585\tchr2\t100\t120\t(CA)n\n")
    rmsk = tmp_path / "rmsk.txt"
    rmsk.write_text("0 1 2 3 4 chr2 100 200 - + AluY SINE Alu 1 2 3 4\n")

    ms = load_microsatellites_for_chrom(microsat, "chr1")
    rm = load_repeatmasker_for_chrom(rmsk, "chr1")

    assert ms is not None and ms.query(110) == (False, None)
    assert rm is not None and rm.query(150) == (False, None)
    assert load_microsatellites_for_chrom(microsat, "chr2").query(110) == (True, "(CA)n")
    assert load_repeatmasker_for_chrom(rmsk, "chr2").query(150) == (True, "SINE")


def test_tmp_dir_is_a_new_subdirectory_and_cleanup_leaves_the_base(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("TMP", raising=False)
    registered = []
    monkeypatch.setattr(io.atexit, "register", lambda fn, *a, **kw: registered.append((fn, a, kw)))

    keep = tmp_path / "ref.fa"
    keep.write_text(">chr1\nACGT\n")
    work = get_hpc_tmp_dir(prefix="chrproc")

    assert work.parent == tmp_path and work.is_dir()

    fn, a, kw = registered[0]
    fn(*a, **kw)
    assert not work.exists()
    # The base and other files in it survive
    assert keep.exists()

