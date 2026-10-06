from __future__ import annotations

import pytest

from src.utility.sample import resolve_patient_id, resolve_sample_id

UNPARSEABLE = "/data/bams/tumor_run7.bam"
SCHEME = "/cohort/C00001/C00001_C00A00001D.bam"


def test_explicit_ids_win_over_the_path():
    assert resolve_sample_id(SCHEME, "SAMPLE001") == "SAMPLE001"
    assert resolve_patient_id(SCHEME, "PATIENT001") == "PATIENT001"


def test_explicit_ids_make_an_unparseable_path_work():
    assert resolve_sample_id(UNPARSEABLE, "SAMPLE001") == "SAMPLE001"
    assert resolve_patient_id(UNPARSEABLE, "PATIENT001", "SAMPLE001") == "PATIENT001"


def test_path_is_the_fallback_when_no_id_is_given():
    assert resolve_sample_id(SCHEME) == "C00A00001D"
    assert resolve_patient_id(SCHEME) == "C00001"


def test_patient_falls_back_to_sample_when_the_path_cannot_be_parsed():
    assert resolve_patient_id(UNPARSEABLE, None, "SAMPLE001") == "SAMPLE001"


def test_unparseable_path_without_an_id_still_fails_loudly():
    with pytest.raises(ValueError, match="Could not find sample ID"):
        resolve_sample_id(UNPARSEABLE)
    with pytest.raises(ValueError, match="Could not find sample ID"):
        resolve_patient_id(UNPARSEABLE)
