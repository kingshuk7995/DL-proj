# Running on Kaggle

To run this project on Kaggle, follow these steps:

## 1. Create a New Notebook
- Set **Accelerator** to **GPU P100** or **GPU T4 x2**.
- Enable **Internet** in the settings.

## 2. Setup Codebase
You can either upload the files or use the following command in a cell to clone (if you have it on GitHub) or create the structure.

Assuming you want to run the training:

```python
# Install dependencies
!pip install -q datasets transformers pyyaml tqdm

# (Optional) If you uploaded the code as a dataset/zip
# !unzip ../input/your-dataset/dl-proj.zip -d .

# Or run directly if you have the files in the current directory
import os
import sys
sys.path.append(os.path.abspath("src"))

# Run training
!python scripts/train.py --config configs/kaggle_wikitext103.yaml
```

## 3. Recommended Config Adjustments for Low Resources
If you still encounter "Out of Memory" (OOM) errors:
- Decrease `batch_size` in the config (e.g., to 4 or 2) and increase `grad_accum` to compensate.
- Decrease `d_model` (e.g., to 256) or `n_layers` (e.g., to 4).
- Use `max_tokens` in the `dataset` section to train on a smaller subset first.

## 4. Logging
Logs will be saved to `/kaggle/working/runs/wikitext103_nopos/train.log`.
Checkpoints will be in `/kaggle/working/runs/wikitext103_nopos/checkpoints/`.
