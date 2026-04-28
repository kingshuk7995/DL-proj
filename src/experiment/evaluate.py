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
def _evaluate_layerwise(model, dataloader: torch.utils.data.DataLoader, device: torch.device):
    model.eval()

    all_layer_feats = None
    all_positions = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        _, hidden_states = model(input_ids, return_hidden_states=True)
        B, T = input_ids.shape
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)  # (B, T)

        if all_layer_feats is None:
            all_layer_feats = [[] for _ in range(len(hidden_states))]

        for l, h in enumerate(hidden_states):
            all_layer_feats[l].append(h.reshape(-1, h.size(-1)).cpu())

        all_positions.append(positions.reshape(-1).cpu())
    all_positions = torch.cat(all_positions, dim=0).float()
    layer_mae = []

    for l in range(len(all_layer_feats)):
        X = torch.cat(all_layer_feats[l], dim=0)
        y = all_positions
        X_aug = torch.cat([X, torch.ones(X.size(0), 1)], dim=1)
        w = torch.linalg.lstsq(X_aug, y).solution
        y_pred = X_aug @ w
        mae = torch.mean(torch.abs(y_pred - y)).item()
        layer_mae.append(mae)

    return layer_mae

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
    evaluation_results = evaluate(model, dataloader, criterion, device)
    evaluation_results['layerwise'] = _evaluate_layerwise(model, dataloader, device)
    return evaluation_results