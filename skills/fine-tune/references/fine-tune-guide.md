# Fine-Tune Guide

## LoRA configuration

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,                    # rank — higher = more parameters, more capacity
    lora_alpha=32,           # scaling factor (typically 2x r)
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",             # "none", "all", or "lora_only"
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
model.print_trainable_parameters()  # verify: ~0.1-1% of total params
```

**Choosing r**: start with r=16. Increase to 32-64 for complex tasks or large datasets.
**target_modules**: for most models, q_proj + v_proj is sufficient. Add k_proj, o_proj, gate_proj, up_proj, down_proj for more capacity.

## QLoRA: 4-bit quantization

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",        # normal float 4-bit
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,   # nested quantization
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)
```

QLoRA = 4-bit base model + LoRA adapters trained in bf16. Reduces memory 4× vs full precision.

## Unsloth: 4x memory reduction

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="mistralai/Mistral-7B-v0.1",
    max_seq_length=2048,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    use_gradient_checkpointing="unsloth",  # saves more memory
)
```

Unsloth rewrites attention kernels in Triton for 2× faster training + 4× memory savings.

## Chat templates

### Alpaca format
```python
def format_alpaca(sample):
    return f"""### Instruction:
{sample['instruction']}

### Input:
{sample.get('input', '')}

### Response:
{sample['output']}"""
```

### ShareGPT format
```python
# Dataset: {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
tokenizer.apply_chat_template(conversations, tokenize=False)
```

### ChatML format
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
{answer}<|im_end|>
```

## SFTTrainer

```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    packing=True,             # concatenate samples to fill context window — 2-4× faster
    formatting_func=format_alpaca,
    args=TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="epoch",
        optim="adamw_8bit",    # memory-efficient optimizer
    ),
)
trainer.train()
```

## DPO training

```python
from trl import DPOTrainer

# Dataset format: {"prompt": "...", "chosen": "...", "rejected": "..."}
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,     # original (frozen) model
    beta=0.1,                # KL penalty coefficient — lower = more deviation allowed
    train_dataset=dpo_dataset,
    tokenizer=tokenizer,
    args=training_args,
)
dpo_trainer.train()
```

**beta**: 0.1 typical. Lower values allow more deviation from the reference. Higher values keep the model closer to the reference (more conservative).

## Merging LoRA

```python
# Merge LoRA weights into base model for inference
model = model.merge_and_unload()
model.save_pretrained("merged_model")
tokenizer.save_pretrained("merged_model")
```

After merging: no LoRA overhead at inference. Model is full-precision.

## GGUF export for local inference

```bash
# Convert merged HuggingFace model to GGUF
python3 convert_hf_to_gguf.py merged_model --outtype f16 --outfile model.gguf

# Quantize to 4-bit
./quantize model.gguf model_q4.gguf Q4_K_M

# Run with llama.cpp
./llama-cli -m model_q4.gguf -p "Your prompt here"
```

## Evaluation

```python
# Perplexity
from transformers import AutoModelForCausalLM
import torch, math

def compute_perplexity(model, tokenizer, texts, max_length=512):
    nlls = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        nlls.append(outputs.loss.item())
    return math.exp(sum(nlls) / len(nlls))

# ROUGE-L
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
results = scorer.score(reference, prediction)
print(f"ROUGE-L: {results['rougeL'].fmeasure:.4f}")
```
