# barenet/config.py
from pathlib import Path
import json

CONFIG_PATH = Path.home() / ".barenet_config.json"
DEFAULT = {"enabled_filters": []}

def load() -> dict:
    if not CONFIG_PATH.exists():
        return DEFAULT.copy()
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return DEFAULT.copy()

def save(cfg: dict):
    try:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass
