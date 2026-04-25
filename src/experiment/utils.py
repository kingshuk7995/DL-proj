from __future__ import annotations

import json
import logging
import math
import random
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_spec: str = "auto") -> torch.device:
    if device_spec != "auto":
        return torch.device(device_spec)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def dataclass_to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: dataclass_to_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [dataclass_to_dict(v) for v in obj]
    return obj


def setup_logging(output_dir: str | Path) -> logging.Logger:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("experiment")
    # Reset handlers if they exist (important for multiple runs in notebooks)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(output_dir / "train.log")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def perplexity(loss: float) -> float:
    if loss >= 50:
        return float("inf")
    return math.exp(loss)


@torch.no_grad()
def top_k_logits(logits: torch.Tensor, k: int | None) -> torch.Tensor:
    if k is None or k <= 0 or k >= logits.size(-1):
        return logits
    vals, idx = torch.topk(logits, k=k)
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(dim=-1, index=idx, src=vals)
    return filtered
