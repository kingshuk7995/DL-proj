from __future__ import annotations
from typing import Literal
import torch
import torch.nn as nn
from .positional import SinusoidalPositionalEncoding, LearnedPositionalEncoding, TransformerBlock


class CausalTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        pos_encoding: Literal["none", "sinusoidal", "learned", "alibi"] = "none",
        tie_weights: bool = True,
        max_len: int = 8192,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pos_encoding = pos_encoding

        self.tok_emb = nn.Embedding(vocab_size, d_model)

        if pos_encoding == "sinusoidal":
            self.pos_emb = SinusoidalPositionalEncoding(d_model, dropout=dropout, max_len=max_len)
        elif pos_encoding == "learned":
            self.pos_emb = LearnedPositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        else:
            self.pos_emb = None

        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    use_alibi=(pos_encoding == "alibi"),
                )
                for _ in range(n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(input_ids)
        if self.pos_emb is not None:
            x = self.pos_emb(x)
        else:
            x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)
