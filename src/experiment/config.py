from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal
import yaml


@dataclass
class DatasetConfig:
    name: str = "wikitext"
    config: str = "wikitext-103-raw-v1"
    tokenizer: str = "gpt2"
    seq_len: int = 512
    stride: int = 256
    max_tokens: Optional[int] = None


@dataclass
class ModelConfig:
    vocab_size: Optional[int] = None
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    pos_encoding: Literal["none", "sinusoidal", "learned", "alibi"] = "none"
    tie_weights: bool = True


@dataclass
class TrainConfig:
    batch_size: int = 16
    num_workers: int = 2
    pin_memory: bool = True
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    grad_accum: int = 1
    amp: bool = True
    optimizer: Literal["adamw", "adam"] = "adamw"
    scheduler: Literal["cosine", "linear_warmup", "none"] = "cosine"
    warmup_steps: int = 500
    log_every: int = 50
    save_best_only: bool = True


@dataclass
class Config:
    seed: int = 42
    device: str = "auto"
    output_dir: str = "runs/nopos_exp"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _merge_dataclass(dc, values: dict):
    from dataclasses import fields, is_dataclass
    if not is_dataclass(dc):
        return values
    for f in fields(dc):
        if f.name in values:
            current = getattr(dc, f.name)
            incoming = values[f.name]
            if hasattr(current, "__dataclass_fields__") and isinstance(incoming, dict):
                _merge_dataclass(current, incoming)
            else:
                setattr(dc, f.name, incoming)
    return dc


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = Config()
    _merge_dataclass(cfg, raw or {})
    return cfg
