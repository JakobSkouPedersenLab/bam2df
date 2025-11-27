from pathlib import Path
import yaml

def load_config(file_path: Path, file_name: str) -> dict:
    with open(Path.joinpath(file_path, file_name), 'r') as stream:
        cfg = yaml.safe_load(stream)
    return cfg