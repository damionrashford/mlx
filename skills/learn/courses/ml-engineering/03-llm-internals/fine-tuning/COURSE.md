# Fine-Tuning LLMs: From General to Specialized

## Learning Objectives

By the end of this lesson, you will:

**Understand:**
- How LoRA achieves parameter-efficient fine-tuning through low-rank decomposition and why it preserves general model knowledge
- The memory and compute tradeoffs between full fine-tuning, LoRA, and QLoRA
- How catastrophic forgetting occurs and what strategies prevent it

**Apply:**
- Configure a QLoRA fine-tuning run using the PEFT and TRL libraries, selecting appropriate rank, target modules, and learning rate
- Prepare instruction-tuning datasets in the correct chat template format

**Analyze:**
- Determine whether a given problem is best solved by prompt engineering, RAG, fine-tuning, or continued pretraining using the decision hierarchy

## Prerequisites

- **Pretraining** -- Understanding of the causal language modeling objective, what a pretrained base model is, and what it can and cannot do before fine-tuning (see [Pretraining](../pretraining/COURSE.md))
- **Training Mechanics** -- Familiarity with loss functions, overfitting, validation, learning rates, and optimizers, since fine-tuning requires careful control of these (see [Training Mechanics](../02-neural-networks/training-mechanics/COURSE.md))

## What Is Fine-Tuning?

Fine-tuning takes a pretrained language model and continues training it on a smaller, task-specific or domain-specific dataset. The model already knows language — fine-tuning teaches it *how you want it to behave*.

Think of it this way: pretraining gives the model general knowledge (like a college education). Fine-tuning is specialized job training. The model already understands grammar, facts, and reasoning. Now you're teaching it to follow instructions, adopt a specific tone, handle your domain's terminology, or excel at a particular task.

```
Pretrained model: "The customer asked about..." → continues writing a news article
Fine-tuned model:  "The customer asked about..." → responds as a helpful shopping assistant
```

## Full Fine-Tuning

Full fine-tuning updates **every parameter** in the model. For a 7B model, that's 7 billion parameters being adjusted.

### When to Use Full Fine-Tuning

- You have a large dataset (>100K examples) and significant compute
- Maximum quality matters more than efficiency
- You're producing a model for wide distribution
- The domain shift from base model is large

### Why Most People Don't Do It

- **Memory**: A 7B model in bf16 = 14 GB for weights. With AdamW optimizer states + gradients, you need ~84 GB just for the optimizer, plus activation memory. That's multiple A100s.
- **Cost**: Multiple GPUs for hours to days.
- **Risk**: Easy to overfit on small datasets. Easy to destroy the model's general capabilities (catastrophic forgetting).
- **Overkill**: For most tasks, PEFT methods achieve 95-99% of full fine-tuning quality at 1% of the cost.

Full fine-tuning is the right call when you're Meta releasing Llama-2-Chat, or when you're training a foundation model for a major product. For everything else, use PEFT.

## Parameter-Efficient Fine-Tuning (PEFT)

PEFT methods freeze most of the pretrained weights and only train a small number of additional or modified parameters. This is the modern default for fine-tuning.

### Why PEFT Works

Neural networks are massively over-parameterized. When you fine-tune for a specific task, the weight changes (deltas) live in a low-dimensional subspace. You don't need to update all 7 billion parameters — you can capture the important changes with far fewer trainable parameters.

This insight is the foundation of LoRA.

## LoRA: Low-Rank Adaptation

LoRA (Hu et al., 2021) is the dominant PEFT method. Here's how it works:

### The Core Idea

For a pretrained weight matrix W (e.g., a 4096x4096 attention projection), instead of fine-tuning W directly, LoRA adds a low-rank decomposition:

```
W' = W + (B @ A)

Where:
  W  = original frozen weights (4096 x 4096 = 16.7M params)
  A  = trainable matrix (4096 x r)      — "down projection"
  B  = trainable matrix (r x 4096)      — "up projection"
  r  = rank (typically 8, 16, 32, 64)   — the key hyperparameter
```

With r=16: trainable params = 4096 * 16 * 2 = 131K (vs 16.7M for full). That's **0.78%** of the original parameters.

### What Rank Means

The rank `r` controls the expressiveness of the adaptation:
- **r=8**: Very efficient. Good for simple tasks like classification or style transfer. ~0.4% of params per adapted layer.
- **r=16**: The sweet spot for most instruction tuning. Good quality/efficiency tradeoff.
- **r=32-64**: For complex tasks requiring significant behavior change. Still much cheaper than full fine-tuning.
- **r=256+**: Approaching full fine-tuning quality. Diminishing returns past r=64 for most tasks.

Think of rank as "how much new information does this task require?" Following instructions in a specific format? r=8 is fine. Learning a new domain's reasoning patterns? You might want r=64.

### Which Layers Get LoRA?

By default, LoRA is applied to the attention projection matrices (Q, K, V, O). But you have choices:

```python
# Common configurations (using PEFT library)

# Minimal — attention query and value projections only
target_modules = ["q_proj", "v_proj"]

# Standard — all attention projections
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Aggressive — attention + MLP layers
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Everything (rare, but sometimes useful)
target_modules = "all-linear"
```

More layers = more trainable parameters = more capacity to learn, but also more compute and more risk of overfitting. Start with attention projections. If quality isn't sufficient, add MLP layers.

### LoRA Alpha

There's a scaling factor `alpha` that controls the magnitude of the LoRA update:

```
W' = W + (alpha / r) * (B @ A)
```

Convention: set `alpha = 2 * r` (so alpha=32 when r=16). This means the effective learning rate for LoRA weights is scaled appropriately. In practice, most people just follow this convention and don't tune alpha independently.

---

### Check Your Understanding

1. LoRA uses a rank-r decomposition (B @ A) instead of updating the full weight matrix W. If W is 4096x4096 and r=16, how many trainable parameters does LoRA add for this one matrix? What percentage of the original matrix is that?
2. Why does LoRA inherently reduce catastrophic forgetting compared to full fine-tuning?
3. You are fine-tuning a model for simple style transfer (making responses more formal). Would you use r=8, r=64, or r=256? Why?

<details>
<summary>Answers</summary>

1. A is 4096x16 and B is 16x4096, so trainable params = 4096 x 16 x 2 = 131,072 (131K). The original matrix has 4096 x 4096 = 16.7M parameters. LoRA adds 131K / 16.7M = 0.78% of the original parameters.
2. In LoRA, the base model weights W are frozen. The general knowledge encoded in W is preserved exactly. Only the small low-rank update (B @ A) is trained, so the model can learn new behavior without overwriting its existing knowledge.
3. r=8. Simple style transfer requires minimal new information -- the model already knows how to write formally, it just needs a nudge to do so consistently. Higher ranks would be wasteful and increase the risk of overfitting on a potentially small style-transfer dataset.

</details>

---

## QLoRA: Fine-Tuning on a Single GPU

QLoRA (Dettmers et al., 2023) made fine-tuning 65B models possible on a single 48GB GPU. The trick: quantize the base model to 4-bit, then add standard LoRA adapters in bf16.

```
Base model (7B):   bf16 = 14 GB   →  4-bit quantized = 3.5 GB
LoRA adapters:     ~50-200 MB (tiny)
Optimizer states:  Only for LoRA params, so ~200-800 MB
Total:             ~4-5 GB  (fits on a consumer GPU!)
```

### How QLoRA Works

1. **Quantize** the base model to 4-bit NormalFloat (NF4) — a data type optimized for normally-distributed neural network weights.
2. **Freeze** these quantized weights.
3. **Add LoRA adapters** in bf16/fp16 (full precision for the small trainable part).
4. **Dequantize on the fly** during forward/backward passes: 4-bit weights to bf16, compute, gradients flow only through LoRA.

The quality loss from 4-bit quantization of the base model is minimal (the model's general knowledge is barely affected), and the LoRA adapters learn the task-specific information in full precision.

### Practical QLoRA Config

```python
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig

# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,  # quantize the quantization constants too
)

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

## Data Preparation

Data quality is the number one determinant of fine-tuning success. Garbage in, garbage out — but amplified, because you're *teaching the model* with this data.

### Instruction Tuning Format

Most fine-tuning uses instruction-response pairs:

```json
{
  "instruction": "Summarize the following product review in one sentence.",
  "input": "I bought these headphones last month and they're amazing. The noise cancellation is top-notch, battery lasts forever, and they're super comfortable for long sessions. Only downside is the case is a bit bulky.",
  "output": "The headphones excel in noise cancellation, battery life, and comfort, with the only drawback being a bulky case."
}
```

### Chat Templates

For conversational models, use the chat template format. This varies by model family:

```
# ChatML format (used by many models)
<|im_start|>system
You are a helpful shopping assistant for an online store.<|im_end|>
<|im_start|>user
What running shoes do you recommend under $100?<|im_end|>
<|im_start|>assistant
Here are my top recommendations for running shoes under $100...<|im_end|>
```

**Critical**: Use the exact chat template that matches your base model. Mismatched templates cause garbage outputs. Check the model's tokenizer config for the template.

### Data Quality Checklist

1. **Consistency**: All examples follow the same format and quality standard.
2. **Diversity**: Cover the range of inputs your model will see in production.
3. **Correctness**: Every output is a response you'd be proud to serve to users.
4. **Edge cases**: Include difficult examples, not just easy ones.
5. **No contradictions**: Don't give conflicting instructions or answers.
6. **Appropriate length**: Responses should match your desired output length in production.

---

### Check Your Understanding

1. QLoRA quantizes the base model to 4-bit but keeps LoRA adapters in bf16. Why is it important that the trainable parameters remain in full precision?
2. What is NF4 (NormalFloat4) and why is it specifically suited for neural network weight quantization?
3. A 7B model in bf16 requires 14 GB. With QLoRA (4-bit base + LoRA adapters + optimizer states), approximately how much GPU memory is needed?

<details>
<summary>Answers</summary>

1. The LoRA adapters are where all the learning happens. Gradients and weight updates require sufficient numerical precision to capture small but meaningful changes. 4-bit precision would introduce too much quantization noise in the gradient computation, preventing effective learning. The base model can tolerate 4-bit because it is frozen and only used for forward/backward pass computation.
2. NF4 is a data type optimized for values that follow a normal (Gaussian) distribution, which neural network weights typically do. It allocates its 16 quantization levels to match the density of the normal distribution, placing more levels near zero (where most weights cluster) and fewer at the tails.
3. Approximately 4-5 GB. The 4-bit base model requires ~3.5 GB, LoRA adapters add ~50-200 MB, and optimizer states for only the LoRA parameters add ~200-800 MB.

</details>

---

## How Much Data Do You Need?

This is the most common question, and the honest answer is "it depends." But here are rules of thumb:

| Task | Examples Needed | Notes |
|------|----------------|-------|
| Style/tone change | 50-200 | "Be more formal" or "respond in bullet points" |
| Simple classification | 200-1000 | Sentiment, intent detection |
| Instruction following | 1K-10K | General instruction tuning |
| Domain-specific Q&A | 5K-50K | Medical, legal, e-commerce |
| Complex reasoning | 10K-100K | Math, multi-step logic |
| New language/modality | 100K+ | Consider continued pretraining instead |

**The quality curve**: Going from 100 to 1000 high-quality examples is a massive improvement. Going from 10K to 100K is marginal for most tasks. **Invest in data quality over quantity.** 500 expert-curated examples often beat 50,000 noisy ones.

## Catastrophic Forgetting

When you fine-tune on task-specific data, the model can "forget" its general capabilities. This is catastrophic forgetting.

**Symptoms:**
- Model excels at your specific task but becomes incoherent on general questions
- The model loses its instruction-following abilities
- Generation quality degrades (repetitive, off-topic)

**Prevention strategies:**

1. **Low learning rate**: Use 1e-5 to 5e-5 for full fine-tuning, 1e-4 to 3e-4 for LoRA. Much lower than pretraining.
2. **Short training**: 1-3 epochs is usually optimal. More epochs = more forgetting.
3. **Mix in general data**: Add 10-20% general instruction data alongside your task-specific data.
4. **LoRA inherently helps**: Since base weights are frozen, general knowledge is preserved. This is a major advantage of PEFT.
5. **Regularization**: Weight decay, dropout in LoRA layers (0.05-0.1).

## Evaluation During Fine-Tuning

### Automated Metrics

- **Training loss**: Should decrease. If it plateaus, you're underfitting. If it drops to near-zero, you're overfitting.
- **Validation loss**: The real signal. Divergence from training loss = overfitting. Stop training when val loss stops improving.
- **Perplexity**: exp(loss). Lower is better. Useful for comparing runs but doesn't directly measure task quality.
- **Task-specific metrics**: BLEU/ROUGE for generation, accuracy for classification, exact match for QA.

### LLM-as-Judge

Increasingly, teams use a stronger LLM (GPT-4, Claude) to evaluate fine-tuned model outputs. This correlates well with human judgment and is much cheaper.

```python
evaluation_prompt = """
Rate the following shopping assistant response on a 1-5 scale for:
1. Helpfulness (does it answer the question?)
2. Accuracy (are the product details correct?)
3. Tone (is it friendly and professional?)

Customer query: {query}
Assistant response: {response}
"""
```

### Human Evaluation

For production-critical applications, there's no substitute for human evaluation. Have domain experts rate outputs on a rubric. This is expensive but gives you the real signal.

## Merging Adapters

After training, you can merge LoRA weights back into the base model:

```python
# Merge LoRA weights into base model
merged_model = model.merge_and_unload()

# Now it's a standard model — no adapter overhead at inference
merged_model.save_pretrained("merged_model")
```

Benefits of merging:
- No inference overhead (no extra matrix multiplications)
- Simpler deployment (single model file)
- Can quantize the merged model for serving

Benefits of keeping separate:
- Swap adapters for different tasks without reloading the base model
- Stack multiple adapters
- Smaller files to distribute (just the adapter, not the full model)

## Multi-Task Fine-Tuning

Train on multiple tasks simultaneously by mixing datasets:

```python
datasets = [
    ("product_qa", 0.4),        # 40% product Q&A
    ("review_summary", 0.2),     # 20% review summarization
    ("recommendation", 0.3),     # 30% product recommendation
    ("general_chat", 0.1),       # 10% general conversation (prevent forgetting)
]
```

The model learns to distinguish tasks from the instruction format. This produces more robust models than single-task fine-tuning.

---

### Check Your Understanding

1. Your fine-tuned shopping assistant gives excellent product recommendations but can no longer answer basic general knowledge questions. What is this phenomenon called and what are two strategies to prevent it?
2. Your training loss drops to near zero after epoch 1 but validation loss starts increasing. What is happening and what should you do?
3. Why is "LLM-as-judge" evaluation increasingly used alongside traditional metrics like BLEU/ROUGE?

<details>
<summary>Answers</summary>

1. Catastrophic forgetting. Prevention strategies include: (a) using LoRA instead of full fine-tuning so base weights are frozen, (b) mixing 10-20% general instruction data into your task-specific training data, (c) using a low learning rate (1e-5 to 5e-5 for full fine-tuning), and (d) training for fewer epochs (1-3).
2. The model is overfitting to the training data. It has memorized the training examples rather than learning generalizable patterns. You should stop training (early stopping), use the checkpoint from epoch 1 or wherever validation loss was lowest, and consider adding regularization (dropout, weight decay) or using more diverse training data.
3. Traditional metrics like BLEU/ROUGE measure surface-level text overlap and correlate poorly with actual response quality for open-ended generation. LLM-as-judge evaluations can assess helpfulness, accuracy, tone, and other qualitative dimensions that better match human judgment, and are much cheaper than human evaluation while correlating well with it.

</details>

---

## Decision Framework: Fine-Tune vs RAG vs Prompt Engineering

This is the most important decision framework for applied ML engineers:

### Start with Prompt Engineering (Always)

Before training anything, try:
1. System prompt engineering with examples
2. Few-shot examples in the prompt
3. Chain-of-thought prompting
4. Structured output formatting

If this gets you to 80%+ quality, **stop here**. Prompt engineering costs nothing, deploys instantly, and works with any model.

### Add RAG When the Model Lacks Knowledge

RAG (Retrieval-Augmented Generation) is the right call when:
- The model needs access to private/current data (product catalogs, internal docs)
- Information changes frequently (inventory, pricing)
- Factual accuracy is critical and you need citations
- You want to avoid hallucination about specific facts

**For an AI personal shopper**: RAG is essential. Product catalogs change constantly. You can't fine-tune every time a merchant adds a product.

### Fine-Tune When the Model Lacks Behavior

Fine-tuning is the right call when:
- You need a specific output format the model struggles with via prompting
- You need a particular tone/personality consistently
- The model needs domain-specific reasoning patterns
- Latency matters (fine-tuned responses need fewer tokens than few-shot prompts)
- Cost matters (shorter prompts = fewer tokens = lower API costs)
- You need to distill a large model's behavior into a smaller, cheaper model

### The Hierarchy

```
1. Prompt engineering    → Free, instant, try first
2. RAG                   → When knowledge is the gap
3. Fine-tuning (LoRA)   → When behavior is the gap
4. Full fine-tuning      → When LoRA isn't enough (rare)
5. Continued pretraining → When the domain is fundamentally different
6. Pretrain from scratch → Almost never (you're not Google)
```

### AI Shopper Example

For an AI personal shopper:
- **Prompt engineering**: Define the shopping assistant persona, output format, safety rules
- **RAG**: Product catalog, reviews, inventory, merchant policies
- **Fine-tuning**: Shopping conversation style, recommendation reasoning, handling returns/complaints, multi-turn dialogue flow
- **NOT pretraining**: E-commerce language is well-covered by existing models

## Practical Walkthrough: Fine-Tuning Llama on Custom Data

Conceptual steps for fine-tuning a model for a shopping assistant:

```python
# 1. Prepare your dataset
# Format: instruction-input-output triples or chat conversations
dataset = load_dataset("your_shopping_conversations")

# 2. Choose your base model
model_name = "meta-llama/Llama-3.1-8B-Instruct"

# 3. Load with quantization (QLoRA)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,  # 4-bit
    device_map="auto",
)

# 4. Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: "trainable params: 13,631,488 || all params: 8,030,261,248 || 0.17%"

# 5. Training arguments
training_args = TrainingArguments(
    output_dir="./shopping-assistant",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,      # effective batch size = 16
    learning_rate=2e-4,
    warmup_steps=100,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    bf16=True,
    lr_scheduler_type="cosine",
)

# 6. Train with SFTTrainer (from TRL library)
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    max_seq_length=2048,
)
trainer.train()

# 7. Save adapter
model.save_pretrained("./shopping-assistant-lora")

# 8. Optionally merge for deployment
merged = model.merge_and_unload()
merged.save_pretrained("./shopping-assistant-merged")
```

## Common Pitfalls

1. **Using the wrong chat template.** Every model family has a specific chat template format (ChatML, Llama format, etc.). Using a mismatched template causes garbage outputs because the model interprets special tokens incorrectly. Always check the model's tokenizer config for the expected format.
2. **Training for too many epochs.** Fine-tuning typically requires only 1-3 epochs. More epochs lead to overfitting and catastrophic forgetting. Monitor validation loss and stop when it plateaus or increases.
3. **Setting learning rate too high.** Fine-tuning learning rates should be much lower than pretraining rates. For LoRA, use 1e-4 to 3e-4. For full fine-tuning, use 1e-5 to 5e-5. Too high a rate will destroy pretrained knowledge.
4. **Prioritizing data quantity over quality.** 500 expert-curated examples often outperform 50,000 noisy ones. Every training example teaches the model a behavior pattern. Low-quality examples teach low-quality behavior.

## Hands-On Exercises

### Exercise 1: QLoRA Setup and Parameter Counting (15 min)

Load a small model (e.g., `TinyLlama/TinyLlama-1.1B-Chat-v1.0`) with QLoRA configuration and explore the trainable parameters.

```python
# Install: pip install transformers peft bitsandbytes accelerate
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import torch

# Load model in 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    quantization_config=bnb_config,
    device_map="auto",
)

# Experiment with different LoRA configs:
# 1. r=8, target_modules=["q_proj", "v_proj"]
# 2. r=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
# 3. r=64, target_modules="all-linear"
# For each, call model.print_trainable_parameters() and compare.
# Question: How does the percentage of trainable parameters change?
```

### Exercise 2: Data Format Inspection (15 min)

Inspect how different chat templates affect tokenization. Load a model's tokenizer and apply its chat template to a conversation, then examine the resulting tokens.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

messages = [
    {"role": "system", "content": "You are a helpful shopping assistant."},
    {"role": "user", "content": "What shoes do you recommend?"},
    {"role": "assistant", "content": "I recommend checking out running shoes from Nike or Brooks."},
]

formatted = tokenizer.apply_chat_template(messages, tokenize=False)
print(formatted)  # See the full template with special tokens

tokens = tokenizer.apply_chat_template(messages, tokenize=True)
print(f"Total tokens: {len(tokens)}")
# Question: How many tokens are "overhead" from the template vs. actual content?
```

## Interview Questions

**Conceptual:**
1. What is LoRA? Explain the low-rank decomposition and why it works.
2. What does the rank `r` control in LoRA? How would you choose it for a new task?
3. Explain catastrophic forgetting. How does LoRA help prevent it compared to full fine-tuning?
4. What is QLoRA and why was it a breakthrough?
5. You have a fine-tuned model that performs well on your test set but poorly in production. What might be wrong?

**Applied:**
6. You're building a customer service chatbot for online merchants. You have 10K labeled conversations. Walk me through your fine-tuning strategy.
7. Your fine-tuned model is producing responses that are too short. What are three things you'd try?
8. A merchant wants their chatbot to always include product links in responses. Would you solve this with fine-tuning, prompting, or post-processing? Why?
9. You have two LoRA adapters — one for English and one for French. How would you serve both from a single base model?
10. Your validation loss starts increasing after epoch 2 but your task-specific metrics keep improving. What's happening and what do you do?

**System Design:**
11. Design a fine-tuning pipeline that automatically retrains when new conversation data arrives. What components do you need? How do you prevent quality regressions?
12. You need to fine-tune for 50 different merchants, each with their own style. How would you architect this efficiently? (Hint: think about adapter management)

**Answer to Q12**: Use a shared base model with per-merchant LoRA adapters. Store adapters in a registry. At inference, load the base model once and dynamically swap the adapter based on the merchant ID. This is memory-efficient (base model loaded once, adapters are tiny) and scales to thousands of merchants. LoRAX or similar serving frameworks support this pattern.

## Summary

This lesson covered how to adapt pretrained models to specific tasks and behaviors. Key takeaways:

- **LoRA** is the default fine-tuning method. It adds low-rank trainable matrices to frozen base weights, achieving 95-99% of full fine-tuning quality at a fraction of the cost and memory.
- **QLoRA** enables fine-tuning large models on a single consumer GPU by quantizing the base model to 4-bit while keeping LoRA adapters in full precision.
- **Data quality is paramount.** Consistent format, correct outputs, diverse inputs, and appropriate response lengths matter far more than dataset size. 500 expert examples often beat 50,000 noisy ones.
- **Catastrophic forgetting** is the main risk of fine-tuning. LoRA, low learning rates, short training runs, and mixing in general data all help prevent it.
- **The decision hierarchy** is: prompt engineering first, then RAG for knowledge gaps, then LoRA fine-tuning for behavior gaps, and full fine-tuning or pretraining only in rare cases.
- **Adapter management** (merging vs. keeping separate) is a deployment decision with practical implications for inference overhead, multi-task serving, and file distribution.

## What's Next

The next lesson, **RLHF and Alignment** (see [RLHF and Alignment](../rlhf-alignment/COURSE.md)), covers how to go beyond supervised fine-tuning to align models with human preferences using reward modeling, DPO, GRPO, and Constitutional AI. Alignment is the stage that transforms a fine-tuned instruction follower into a helpful, honest, and harmless assistant.
