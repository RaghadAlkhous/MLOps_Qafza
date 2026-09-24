from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config() -> dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Convert relative paths to absolute paths based on PROJECT_ROOT
    for key, value in cfg.get("paths", {}).items():
        p = Path(value)
        if not p.is_absolute():
            cfg["paths"][key] = str(PROJECT_ROOT / p)

    return cfg


CONFIG = load_config()
