from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.utility.config import validate_bam_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_shipped_config_is_valid():
    cfg = json.loads((REPO_ROOT / "config.json").read_text())
    validate_bam_config(cfg["bam"])


@pytest.mark.parametrize(
    "bam_config, message",
    [
        ({"min_base_quality": 30, "min_bq": 37}, "no longer supported"),
        ({}, "missing required key"),
        ({"min_base_quality": -1}, "non-negative integer"),
        ({"min_base_quality": 30.0}, "non-negative integer"),
        ({"min_base_quality": True}, "non-negative integer"),
    ],
)
def test_invalid_base_quality_settings_fail(bam_config, message):
    with pytest.raises(ValueError, match=message):
        validate_bam_config(bam_config)
