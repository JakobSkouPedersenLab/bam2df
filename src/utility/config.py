from pathlib import Path
import yaml

def load_config(file_path: Path, file_name: str) -> dict:
    with open(Path.joinpath(file_path, file_name), 'r') as stream:
        cfg = yaml.safe_load(stream)
    return cfg

def validate_bam_config(bam_config: dict) -> None:
    """
    Fail early on settings the extractor would otherwise read wrongly.

    `min_base_quality` is required: both mates must reach it at a position for that
    position to be scanned. The former key `min_bq` is rejected rather than ignored.
    """
    if "min_bq" in bam_config:
        raise ValueError("config key 'bam.min_bq' is no longer supported; use 'bam.min_base_quality'")
    if "min_base_quality" not in bam_config:
        raise ValueError("config is missing required key 'bam.min_base_quality'")
    min_bq = bam_config["min_base_quality"]
    if isinstance(min_bq, bool) or not isinstance(min_bq, int) or min_bq < 0:
        raise ValueError(f"'bam.min_base_quality' must be a non-negative integer, got {min_bq!r}")