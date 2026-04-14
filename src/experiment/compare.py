from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict

from .config import Config
from .train import train


def run_comparison(base_cfg: Config, pos_types=("none", "sinusoidal", "learned", "alibi")) -> Dict:
    results = {}
    for pos in pos_types:
        cfg = deepcopy(base_cfg)
        cfg.model.pos_encoding = pos
        cfg.output_dir = str(Path(base_cfg.output_dir) / pos)
        results[pos] = train(cfg)
    return results
