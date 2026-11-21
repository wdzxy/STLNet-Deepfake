import json
from pathlib import Path


def load_config(filename):
    path = Path(filename)
    with path.open(encoding="utf-8-sig") as handle:
        if path.suffix.lower() in {".yaml", ".yml"}:
            import yaml
            config = yaml.safe_load(handle)
        else:
            config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    return config
