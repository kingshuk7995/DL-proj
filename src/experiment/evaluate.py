from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .config import Config
from .data import build_dataloaders
from .model import CausalTransformerLM
from .train import evaluate
from .utils import get_device


def load_checkpoint(path: str | Path, device: torch.device):
    return torch.load(path, map_location=device)


@torch.no_grad()
def evaluate_checkpoint(cfg: Config, checkpoint_path: str | Path, split: str = "test"):
    device = get_device(cfg.device)
    tokenizer, train_dl, val_dl, test_dl = build_dataloaders(cfg)
    cfg.model.vocab_size = tokenizer.vocab_size
    model = CausalTransformerLM(
        vocab_size=cfg.model.vocab_size,
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        d_ff=cfg.model.d_ff,
        dropout=cfg.model.dropout,
        pos_encoding=cfg.model.pos_encoding,
        tie_weights=cfg.model.tie_weights,
    ).to(device)

    ckpt = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(ckpt["model_state"], strict=True)

    criterion = nn.CrossEntropyLoss()
    dataloader = {"train": train_dl, "val": val_dl, "test": test_dl}[split]
    return evaluate(model, dataloader, criterion, device)
