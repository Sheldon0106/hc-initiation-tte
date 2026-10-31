"""Path configuration only; no patient files are read at import time."""
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def settings():
    p = Path(os.environ.get('HC_TTE_CONFIG', str(REPO / 'config.local.json')))
    if p.exists():
        with p.open(encoding='utf-8-sig') as stream:
            return json.load(stream)
    return {}


def configured_path(key, default):
    p = Path(settings().get(key, default))
    return p if p.is_absolute() else (REPO / p).resolve()
