# Phishing Detection with Fine-Tuned SLMs

Fine-tuning [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) to classify webpages as **phishing** or **benign** based on their URL and cleaned HTML content.

## Dataset

[phreshphish/phreshphish](https://huggingface.co/datasets/phreshphish/phreshphish) — 5 000 samples streamed, filtered to English-only, HTML cleaned to phishing-relevant tags (`title`, `input`, `iframe`), capped at 5 000 chars, balanced classes.

## Pipeline

1. **Data loading & EDA** — Stream from HuggingFace, clean HTML with `lxml` (strip scripts/styles, keep only structural tags + small attribute allowlist), filter to English, analyse URL signals (TLD distribution, URL length, path depth)
2. **Class balancing** — Downsample the majority class to match the minority, split 80/10/10
3. **Model & LoRA setup** — Load Qwen3-0.6B in 4-bit via [Unsloth](https://github.com/unslothai/unsloth), attach rank-16 LoRA to all attention + MLP projections
4. **Prompt design** — Chat-templated classification prompt with XML-tagged output (`<prediction>phish</prediction>`)
5. **Baseline eval** — Zero-shot performance on the validation set before any fine-tuning
6. **SFT training** — 100 steps with sequence packing, 8-bit AdamW, constant LR (5e-4) with warmup, gradient accumulation 4
7. **Evaluation** — Accuracy, precision, recall, F1, and confusion matrix on validation and held-out test sets

## Setup

```bash
pip install unsloth lxml 
```

Requires a CUDA GPU (T4 or better). Run `main_t4_full.ipynb` end-to-end.
