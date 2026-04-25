# Phishing Detection with Fine-Tuned LLM

Fine-tuning [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) to classify webpages as **phishing** or **benign** based on their URL and cleaned HTML content.

## Dataset

[phreshphish/phreshphish](https://huggingface.co/datasets/phreshphish/phreshphish) — English-only subset, balanced classes, HTML stripped to structural tags only.

## Pipeline

1. **Data prep** — Filter to English, clean HTML (keep forms/links/meta), cap length at 20k chars, balance classes, split 80/10/10
2. **Baseline eval** — Zero-shot classification with the base model
3. **Fine-tuning** — SFT with `SFTTrainer` (AdamW fused, lr=3e-4, 75 steps, bf16)
4. **Evaluation** — Accuracy, precision, recall, F1 on validation and test sets

## Setup

```bash
pip install transformers datasets matplotlib seaborn scikit-learn beautifulsoup4 trl pandas torch liger-kernel peft
```

Requires a CUDA GPU. Run `main.ipynb` end-to-end.
