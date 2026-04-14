from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import GPT2TokenizerFast

from .config import Config


class CausalLMDataset(Dataset):
    """
    Creates overlapping fixed-length training examples from a token stream.
    Returns (x, y) where y is x shifted by one token.
    """

    def __init__(self, token_ids: torch.Tensor, seq_len: int, stride: int):
        super().__init__()
        if token_ids.dim() != 1:
            raise ValueError("token_ids must be a 1D tensor")
        self.token_ids = token_ids
        self.seq_len = seq_len
        self.stride = stride
        self.starts = []
        max_start = len(token_ids) - (seq_len + 1)
        for s in range(0, max_start + 1, stride):
            self.starts.append(s)
        if not self.starts:
            raise ValueError("Not enough tokens to create one sequence")

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        s = self.starts[idx]
        chunk = self.token_ids[s : s + self.seq_len + 1]
        x = chunk[:-1].clone()
        y = chunk[1:].clone()
        return x, y


def load_tokenizer(name: str) -> GPT2TokenizerFast:
    tokenizer = GPT2TokenizerFast.from_pretrained(name)
    if tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.sep_token or ""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _encode_text(tokenizer: GPT2TokenizerFast, raw_split) -> torch.Tensor:
    texts = [t.strip() for t in raw_split["text"] if t and t.strip()]
    if not texts:
        raise ValueError("No usable text found in the dataset split")
    
    all_ids = []

    chunk_size = 1000
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i : i + chunk_size]
        encoded = tokenizer(chunk, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)
        for ids in encoded["input_ids"]:
            all_ids.extend(ids)
            all_ids.append(tokenizer.eos_token_id)
            
    if all_ids:
        all_ids.pop()
        
    return torch.tensor(all_ids, dtype=torch.long)


def build_datasets(cfg: Config):
    tokenizer = load_tokenizer(cfg.dataset.tokenizer)

    raw_train = load_dataset(cfg.dataset.name, cfg.dataset.config, split="train")
    raw_val = load_dataset(cfg.dataset.name, cfg.dataset.config, split="validation")
    raw_test = load_dataset(cfg.dataset.name, cfg.dataset.config, split="test")

    train_ids = _encode_text(tokenizer, raw_train)
    val_ids = _encode_text(tokenizer, raw_val)
    test_ids = _encode_text(tokenizer, raw_test)

    if cfg.dataset.max_tokens is not None:
        train_ids = train_ids[: cfg.dataset.max_tokens]

    train_ds = CausalLMDataset(train_ids, cfg.dataset.seq_len, cfg.dataset.stride)
    val_ds = CausalLMDataset(val_ids, cfg.dataset.seq_len, cfg.dataset.seq_len)
    test_ds = CausalLMDataset(test_ids, cfg.dataset.seq_len, cfg.dataset.seq_len)

    return tokenizer, train_ds, val_ds, test_ds


def build_dataloaders(cfg: Config):
    tokenizer, train_ds, val_ds, test_ds = build_datasets(cfg)

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        drop_last=False,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        drop_last=False,
    )

    return tokenizer, train_dl, val_dl, test_dl
