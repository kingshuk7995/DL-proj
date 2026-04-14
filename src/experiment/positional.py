from __future__ import annotations
import math
import torch
import torch.nn as nn


def alibi_slopes(n_heads: int) -> torch.Tensor:
    def get_slopes_power_of_2(n):
        start = 2.0 ** (-2.0 ** (-(math.log2(n) - 3)))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]

    if math.log2(n_heads).is_integer():
        slopes = get_slopes_power_of_2(n_heads)
    else:
        closest_pow2 = 2 ** math.floor(math.log2(n_heads))
        slopes = get_slopes_power_of_2(closest_pow2)
        extra = get_slopes_power_of_2(2 * closest_pow2)[0::2][: n_heads - closest_pow2]
        slopes.extend(extra)
    return torch.tensor(slopes, dtype=torch.float32)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 8192):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model, dtype=torch.float32)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.size(1)
        x = x + self.pe[:t].transpose(0, 1)
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8192, dropout: float = 0.1):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        pos = torch.arange(t, device=x.device).unsqueeze(0).expand(b, t)
        return self.dropout(x + self.pos_emb(pos))


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, use_alibi: bool = False):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_alibi = use_alibi

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        if use_alibi:
            self.register_buffer("slopes", alibi_slopes(n_heads), persistent=False)

    def _alibi_bias(self, t: int, device, dtype):
        i = torch.arange(t, device=device)
        j = torch.arange(t, device=device)
        dist = (i[None, :] - j[:, None]).clamp(min=0).to(dtype)
        slopes = self.slopes.to(device=device, dtype=dtype).view(1, self.n_heads, 1, 1)
        return -slopes * dist.view(1, 1, t, t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(causal_mask.view(1, 1, t, t), float("-inf"))

        if self.use_alibi:
            att = att + self._alibi_bias(t, x.device, att.dtype)

        att = torch.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        y = self.out(y)
        return self.resid_drop(y)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, use_alibi: bool = False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, use_alibi=use_alibi)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
