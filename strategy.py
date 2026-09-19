"""Submission entry point: strictly sector-neutral carry core plus a cost-gated,
within-sector reversal overlay. The logic lives in src/strategy_core.py; this file
only loads the frozen configuration from artifacts/params.json.
"""
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.strategy_core import Allocator, Config  # noqa: E402

_PARAMS = _ROOT / "artifacts" / "params.json"
_config = Config(**json.loads(_PARAMS.read_text())["config"]) if _PARAMS.exists() else Config()
_allocator = Allocator(_config)


def generate_positions(history, current_date):
    return _allocator.decide(history, current_date)
