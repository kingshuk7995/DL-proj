from __future__ import annotations

import json
import math
import random
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
        return asdict(obj)
    return obj


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
