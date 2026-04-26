# NoPos Transformer LM

A multi-file PyTorch project that reproduces the paper *Transformer Language Models without Positional Encodings Still Learn Positional Information* in a clean training/evaluation layout.

Default dataset: **WikiText-103** (`wikitext-103-raw-v1`), which is much larger than WikiText-2 and is the right starting point when the smaller dataset overfits too fast.

Implemented positional variants:
- `none` (NoPos)
- `sinusoidal`
- `learned`
- `alibi`

## Run

```bash
uv lock
uv run scripts/train.py --config configs/wikitext103.yaml
uv run scripts/evaluate.py --config configs/wikitext103.yaml --checkpoint runs/exp/checkpoints/best.pt --split test
uv run scripts/generate.py --config configs/wikitext103.yaml --checkpoint runs/exp/checkpoints/best.pt --prompt "The theory of"
```
- For running on Kaggle, see [KAGGLE_GUIDE.md](KAGGLE_GUIDE.md).

## Notes

- This project is designed for causal language modeling.
- The code is structured to make it easy to swap:
  - dataset
  - positional encoding
  - optimizer
  - model size
- For a paper-style run, increase model width/depth and batch size as your hardware allows.

For checkpoints, download from here [drive link](https://drive.google.com/drive/folders/1Mp3hQ5uDDmmrYxwZuk2HJacZp-2Rbbmx?usp=sharing) and paste to ./runs/wikitext103_nopos/checkpoints