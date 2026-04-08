---
name: fine-tune
description: >
  Fine-tune language models with LoRA, QLoRA, or full fine-tuning. Covers unsloth
  (4x memory reduction), PEFT, trl SFTTrainer, DPO, instruction tuning with chat
  templates, dataset preparation, and evaluation. Use when fine-tuning any HuggingFace
  model on custom data.
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
argument-hint: base model name or path, dataset path (e.g. "mistralai/Mistral-7B-v0.1 data/train.jsonl")
---

# Fine-Tune Skill

Fine-tune language models efficiently with LoRA, QLoRA, or unsloth.

## Quick start

```bash
# Prepare dataset
python3 scripts/prepare_dataset.py data/raw.csv --format alpaca --output data/train.jsonl

# Fine-tune with LoRA (unsloth)
python3 -c "
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained('mistralai/Mistral-7B-v0.1', load_in_4bit=True)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=32, target_modules=['q_proj','v_proj'])
# ... training loop
"
```

## Workflow

1. **Prepare dataset** — convert to alpaca/sharegpt format via `scripts/prepare_dataset.py`
2. **Choose method** — LoRA (parameter-efficient), QLoRA (memory-efficient), full (highest quality)
3. **Configure LoRA** — set r, lora_alpha, target_modules
4. **Train** — SFTTrainer with packing=True for efficiency
5. **Merge** — `model.merge_and_unload()` for deployment
6. **Evaluate** — perplexity, ROUGE-L, task benchmarks

See `references/fine-tune-guide.md` for complete LoRA/QLoRA/unsloth documentation.
