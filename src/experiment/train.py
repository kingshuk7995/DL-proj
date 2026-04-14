from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from tqdm import tqdm

from .config import Config
from .data import build_dataloaders
from .model import CausalTransformerLM
from .utils import ensure_dir, seed_everything, get_device, save_json, perplexity


def _batch_to_device(batch, device):
    x, y = batch
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def _loss(logits: torch.Tensor, targets: torch.Tensor, criterion):
    b, t, v = logits.shape
    return criterion(logits.reshape(b * t, v), targets.reshape(b * t))


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in dataloader:
        x, y = _batch_to_device(batch, device)
        logits = model(x)
        loss = _loss(logits, y, criterion)
        n = y.numel()
        total_loss += loss.item() * n
        total_tokens += n
    avg = total_loss / max(total_tokens, 1)
    return {"loss": avg, "ppl": perplexity(avg)}


def build_optimizer(cfg: Config, model: nn.Module):
    if cfg.train.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    return torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)


def build_scheduler(cfg: Config, optimizer, steps_per_epoch: int):
    total_steps = max(cfg.train.epochs * steps_per_epoch, 1)
    warmup = min(cfg.train.warmup_steps, max(total_steps // 2, 0))

    if cfg.train.scheduler == "none":
        return None

    def lr_lambda(step):
        if warmup > 0 and step < warmup:
            return max(step / warmup, 1e-8)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        progress = min(max(progress, 0.0), 1.0)
        if cfg.train.scheduler == "linear_warmup":
            return max(1.0 - progress, 0.0)
        import math
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(cfg: Config) -> Dict:
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    outdir = ensure_dir(cfg.output_dir)
    ckpt_dir = ensure_dir(outdir / "checkpoints")

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

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer, steps_per_epoch=len(train_dl))
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.train.amp and device.type == "cuda"))

    best_val = float("inf")
    history = []
    global_step = 0

    save_json(outdir / "config.json", {
        "seed": cfg.seed,
        "device": cfg.device,
        "output_dir": cfg.output_dir,
        "dataset": cfg.dataset.__dict__,
        "model": cfg.model.__dict__,
        "train": cfg.train.__dict__,
    })

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        running_loss = 0.0
        running_tokens = 0

        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{cfg.train.epochs}", leave=False)
        for batch in pbar:
            x, y = _batch_to_device(batch, device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=(cfg.train.amp and device.type == "cuda")):
                logits = model(x)
                loss = _loss(logits, y, criterion)

            scaler.scale(loss).backward()
            if cfg.train.grad_clip is not None and cfg.train.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            if scheduler is not None:
                scheduler.step()

            n = y.numel()
            running_loss += loss.item() * n
            running_tokens += n
            global_step += 1

            if global_step % cfg.train.log_every == 0:
                avg = running_loss / max(running_tokens, 1)
                pbar.set_postfix(train_loss=f"{avg:.4f}", train_ppl=f"{perplexity(avg):.2f}")

        train_loss = running_loss / max(running_tokens, 1)
        train_metrics = {"loss": train_loss, "ppl": perplexity(train_loss)}
        val_metrics = evaluate(model, val_dl, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_ppl": train_metrics["ppl"],
            "val_loss": val_metrics["loss"],
            "val_ppl": val_metrics["ppl"],
        }
        history.append(row)
        print(row)

        last_path = ckpt_dir / "last.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "cfg": {
                    "dataset": cfg.dataset.__dict__,
                    "model": cfg.model.__dict__,
                    "train": cfg.train.__dict__,
                },
            },
            last_path,
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_path = ckpt_dir / "best.pt"
            torch.save(torch.load(last_path, map_location="cpu"), best_path)

    test_metrics = evaluate(model, test_dl, criterion, device)

    save_json(outdir / "history.json", history)
    save_json(outdir / "test_metrics.json", test_metrics)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "history": history,
        "test_metrics": test_metrics,
        "output_dir": str(outdir),
    }
