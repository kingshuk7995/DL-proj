from __future__ import annotations
from typing import Optional
import torch
from .utils import top_k_logits


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = 50,
    device: Optional[torch.device] = None,
) -> str:
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    for _ in range(max_new_tokens):
        logits = model(input_ids)
        next_logits = logits[:, -1, :].squeeze(0) / max(temperature, 1e-6)
        next_logits = top_k_logits(next_logits, top_k)
        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).view(1, 1)
        input_ids = torch.cat([input_ids, next_id], dim=1)

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)
