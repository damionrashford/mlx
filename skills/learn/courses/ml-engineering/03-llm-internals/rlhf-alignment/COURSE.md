# RLHF and Alignment: Making Models Useful and Safe

## Learning Objectives

By the end of this lesson, you will:

**Understand:**
- The three-stage alignment pipeline (pretraining, SFT, preference optimization) and the distinct purpose of each stage
- How DPO eliminates the need for a separate reward model by rearranging the RLHF objective into a closed-form supervised loss
- Why GRPO is particularly effective for tasks with verifiable rewards (math, code) and how it differs from DPO and PPO

**Apply:**
- Set up a DPO training run using the TRL library with appropriate beta, learning rate, and preference data format
- Design a preference data collection pipeline with clear annotation rubrics

**Analyze:**
- Select the appropriate alignment method (SFT, DPO, GRPO, PPO, or CAI) for a given use case based on available data, reward signal type, and compute constraints

## Prerequisites

- **Fine-Tuning** -- Understanding of supervised fine-tuning, LoRA, and how instruction-tuning data is formatted, since SFT is the foundation that alignment builds upon (see [Fine-Tuning](../fine-tuning/COURSE.md))
- **Probability and Statistics** -- Familiarity with KL divergence is important for understanding the constraint term in both PPO and DPO objectives that prevents policy models from diverging too far from the reference model (see [Probability and Statistics](../01-foundations/probability-statistics/COURSE.md))

## Why Alignment Matters

A pretrained language model is an autocomplete engine. Given "The president of the United States", it might continue with a Wikipedia-style paragraph, a news article excerpt, or fan fiction — whatever pattern matches its training data. It doesn't *try* to be helpful. It doesn't *try* to avoid harm. It just predicts likely next tokens.

Alignment is the process of shaping model behavior so it:
1. **Follows instructions** instead of just completing text
2. **Is helpful** — gives useful, accurate answers
3. **Is honest** — doesn't fabricate information (or admits uncertainty)
4. **Is harmless** — refuses dangerous requests, avoids bias

The gap between "raw intelligence" (pretraining) and "useful assistant" (aligned model) is enormous. GPT-3's base model was impressive but nearly unusable as a product. InstructGPT (the aligned version) — trained with vastly less compute — was the one people actually wanted to use.

## The Alignment Pipeline

Modern LLMs go through a three-stage pipeline:

```
Stage 1: Pretraining (trillions of tokens, CLM objective)
    |
    v
Stage 2: Supervised Fine-Tuning / SFT (thousands of instruction-response pairs)
    |
    v
Stage 3: Preference Optimization (RLHF, DPO, or variants)
```

Each stage serves a distinct purpose. Let's examine each.

## Stage 2: Supervised Fine-Tuning (SFT)

SFT bridges the gap between "text completer" and "instruction follower." You train the model on high-quality (instruction, response) pairs written by humans.

### What SFT Data Looks Like

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing to a 10-year-old."},
    {"role": "assistant", "content": "Imagine you have a magic coin that can be both heads AND tails at the same time..."}
  ]
}
```

### Key Properties of SFT

- **Dataset size**: Typically 10K-100K high-quality examples. LIMA (Meta, 2023) showed that just 1,000 carefully curated examples can produce a strong instruction-following model — quality over quantity.
- **Data diversity**: Cover many task types — Q&A, summarization, coding, math, creative writing, safety refusals.
- **Response quality**: Every response should be a "gold standard" answer. If your SFT data contains mediocre responses, your model learns to be mediocre.
- **Multi-turn conversations**: Include multi-turn dialogues, not just single exchanges. Models need to learn to maintain context.

### SFT Transforms the Model

Before SFT:
```
User: "What's the capital of France?"
Model: "What's the capital of Germany? What's the capital of Italy?" (continues the pattern)
```

After SFT:
```
User: "What's the capital of France?"
Model: "The capital of France is Paris." (answers the question)
```

SFT alone gets you surprisingly far. Many open-source models are SFT-only and perform well. But SFT has a ceiling: the model can only be as good as the best responses in your training data. Preference optimization pushes past this ceiling.

## Reward Modeling

Before we can do RLHF, we need a reward model — a model that scores outputs by how much humans would prefer them.

### How Reward Models Are Trained

1. **Generate responses**: For each prompt, generate 2+ candidate responses from the SFT model (or varying-quality models).
2. **Human ranking**: Annotators rank the responses from best to worst.
3. **Train the reward model**: A model (often initialized from the SFT model) that takes (prompt, response) and outputs a scalar score. Trained with a pairwise ranking loss.

```
Prompt: "How do I make pasta?"
Response A: "Boil water, add salt, cook pasta 8-10 min, drain." (rated: good)
Response B: "Pasta is a type of food." (rated: bad)

Reward model learns: score(A) > score(B)
```

The pairwise loss (Bradley-Terry model):

```
L = -log(sigmoid(r(x, y_preferred) - r(x, y_rejected)))
```

This pushes the reward model to assign higher scores to preferred responses and lower scores to rejected ones.

### Reward Model Quality

The reward model is the bottleneck of the entire RLHF pipeline. If it's wrong about what "good" means, the whole system optimizes for the wrong thing. Common issues:

- **Verbosity bias**: Reward models tend to prefer longer responses, even when shorter is better.
- **Sycophancy**: Models that agree with the user score higher, even when the user is wrong.
- **Format hacking**: The model learns superficial patterns that score well (markdown formatting, bullet points) without substance.

---

### Check Your Understanding

1. A pretrained model responds to "What's the capital of France?" by continuing with "What's the capital of Germany? What's the capital of Italy?" instead of answering. Why does this happen, and what stage of the alignment pipeline fixes it?
2. The LIMA paper showed that 1,000 carefully curated SFT examples can produce a strong instruction-following model. What does this imply about the relative importance of data quality vs. quantity for SFT?
3. A reward model is trained using the Bradley-Terry pairwise loss. What does this loss function optimize for, and what is a common failure mode of reward models?

<details>
<summary>Answers</summary>

1. The pretrained model was trained to predict likely next tokens, not to answer questions. The pattern of listing similar questions is a common text pattern on the internet. SFT (Stage 2) fixes this by training on (instruction, response) pairs that teach the model to answer questions directly.
2. It implies that data quality is far more important than quantity for SFT. Each example serves as a template for the model's behavior. A small number of high-quality, diverse examples can teach the model the general pattern of instruction following, which then generalizes to novel instructions.
3. The Bradley-Terry loss optimizes the reward model to assign higher scalar scores to preferred responses than to rejected responses. Common failure modes include: verbosity bias (preferring longer responses), sycophancy (preferring responses that agree with the user), and format hacking (preferring well-formatted responses regardless of substance).

</details>

---

## RLHF with PPO

Reinforcement Learning from Human Feedback (RLHF) uses the reward model to further optimize the SFT model.

### The Setup

- **Policy model**: The SFT model being optimized (generates responses)
- **Reward model**: Scores responses (frozen during RLHF)
- **Reference model**: A frozen copy of the SFT model (prevents the policy from diverging too far)
- **Value model**: Estimates expected future reward (standard RL component)

That's **four models** in memory simultaneously. This is why RLHF is expensive and complex.

### The PPO Objective

```
L = E[reward(response) - beta * KL(policy || reference)]
```

The reward term pushes the model toward high-scoring responses. The KL divergence term prevents it from straying too far from the SFT model (which prevents reward hacking and mode collapse).

`beta` is the KL penalty coefficient — a critical hyperparameter. Too low: model hacks the reward. Too high: model barely changes from SFT.

### The PPO Training Loop

```
for each batch of prompts:
    1. Generate responses using current policy
    2. Score responses with reward model
    3. Compute advantage estimates (how much better/worse than expected)
    4. Update policy to increase probability of high-reward responses
    5. Clip updates to prevent too-large policy changes (PPO's key innovation)
    6. Update value model
```

### Problems with RLHF/PPO

1. **Complexity**: Four models, careful hyperparameter tuning, unstable training.
2. **Reward hacking**: The model finds exploits in the reward model. Example: generating extremely verbose responses because the reward model has a verbosity bias.
3. **Mode collapse**: The model converges to a narrow set of "safe" responses that score well but lack diversity.
4. **Computational cost**: 4 models in memory. Training is 2-4x more expensive than SFT alone.
5. **Reproducibility**: Small changes in hyperparameters lead to very different outcomes. Hard to debug.

These problems led researchers to seek simpler alternatives. Enter DPO.

## DPO: Direct Preference Optimization

DPO (Rafailov et al., 2023) was a breakthrough in simplicity. The key insight: you can optimize directly on human preferences without training a separate reward model.

### The DPO Insight

The authors showed that the optimal RLHF objective can be rearranged into a closed-form loss that depends only on the policy model and the preference data:

```
L_DPO = -log(sigmoid(beta * (log(pi(y_w|x)/pi_ref(y_w|x)) - log(pi(y_l|x)/pi_ref(y_l|x)))))

Where:
  y_w = preferred (winning) response
  y_l = rejected (losing) response
  pi = current policy
  pi_ref = reference model (frozen SFT model)
  beta = temperature parameter
```

In plain English: increase the probability of preferred responses and decrease the probability of rejected responses, relative to the reference model.

### Why DPO Is Better (Usually)

| Aspect | RLHF/PPO | DPO |
|--------|----------|-----|
| Models needed | 4 (policy, reward, reference, value) | 2 (policy, reference) |
| Training stability | Fragile, requires careful tuning | Stable, standard supervised training |
| Hyperparameters | Many (KL coeff, clip range, learning rate, etc.) | Few (beta, learning rate) |
| Compute cost | High | ~Same as SFT |
| Implementation | Complex (RL loop) | Simple (standard loss function) |
| Quality | State of the art | Comparable to RLHF, sometimes better |

### DPO Data Format

```json
{
  "prompt": "How do I improve my website's loading speed?",
  "chosen": "Here are the most impactful optimizations:\n1. Compress images (use WebP format)\n2. Enable browser caching\n3. Minimize CSS/JS files\n4. Use a CDN\n5. Lazy-load below-fold content",
  "rejected": "You should make your website faster. There are many ways to do this. Fast websites are better than slow websites. You can try different things to make it faster."
}
```

You need pairs of (preferred, rejected) responses for each prompt. These can come from:
- Human annotators directly writing/selecting responses
- Generating multiple responses from the model and having humans rank them
- Using a strong model (GPT-4) as a judge to create synthetic preferences

### DPO Training Code (Conceptual)

```python
from trl import DPOTrainer, DPOConfig

training_args = DPOConfig(
    output_dir="./aligned-model",
    beta=0.1,                       # KL penalty strength
    learning_rate=5e-7,             # Very low — we're refining, not retraining
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    bf16=True,
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,            # Frozen SFT model
    args=training_args,
    train_dataset=preference_data,
    tokenizer=tokenizer,
)
trainer.train()
```

## GRPO: Group Relative Policy Optimization

GRPO (Shao et al., 2024, from DeepSeek) is a newer approach that simplifies optimization further, especially for reasoning tasks.

### The Key Difference

Instead of needing explicit preference pairs (DPO) or a separate reward model (PPO), GRPO:
1. Generates a **group** of responses for each prompt (e.g., 8-16 responses)
2. Scores them with a simple, often rule-based reward (e.g., "did the math answer match the correct answer?")
3. Uses the group statistics to compute relative advantages — no value model needed

```
For prompt P, generate responses [r1, r2, ..., r8]
Compute rewards: [0.2, 0.8, 0.5, 0.9, 0.1, 0.7, 0.3, 0.6]
Normalize within group: advantages = (rewards - mean) / std
Update policy: increase probability of above-average responses
```

### Why GRPO Matters

- **No reward model needed**: The reward can be a simple function (correct/incorrect, format compliance, etc.)
- **No value model needed**: Advantages are estimated from the group, not a learned value function
- **Works great for verifiable tasks**: Math, code, structured outputs — anything where you can programmatically check correctness
- **Key to DeepSeek-R1's success**: The reasoning capabilities of DeepSeek-R1 were largely attributed to GRPO training on math and code tasks

### Limitation

GRPO works best when you have a reliable automated reward signal. For subjective tasks (creative writing, open-ended conversation), DPO with human preferences may still be better.

---

### Check Your Understanding

1. The PPO objective includes a KL divergence penalty: `L = E[reward(response) - beta * KL(policy || reference)]`. What happens if beta is set too low? Too high?
2. DPO eliminates the need for a reward model and a value model. What data format does DPO require instead, and where can this data come from?
3. GRPO generates a group of responses and uses within-group statistics to compute advantages. Why does this eliminate the need for a value model?

<details>
<summary>Answers</summary>

1. If beta is too low, the KL penalty is weak, allowing the policy to deviate far from the reference model. This leads to reward hacking -- the model exploits weaknesses in the reward model rather than genuinely improving. If beta is too high, the policy barely changes from the SFT model, and the RLHF training has minimal effect.
2. DPO requires preference pairs: for each prompt, a preferred (chosen) response and a rejected response. These can come from human annotators ranking model outputs, from generating multiple responses and having humans select the best, or from using a strong model (like GPT-4) as a judge to create synthetic preferences.
3. In standard RL (PPO), the value model estimates expected future reward to compute advantages (how much better/worse than expected an action was). GRPO instead estimates this directly from the group: it normalizes rewards within each group (subtracting the mean, dividing by std) to get relative advantages. The group itself serves as the baseline, eliminating the need for a learned value function.

</details>

---

## Constitutional AI (Anthropic)

Constitutional AI (CAI) takes a different approach: instead of relying entirely on human annotations, the model critiques and revises its own outputs using a set of principles (a "constitution").

### The Process

1. **Generate**: Model generates a response to a potentially harmful prompt.
2. **Critique**: Model evaluates its own response against principles: "Is this response harmful?" "Does it respect privacy?" "Is it honest?"
3. **Revise**: Model generates an improved response based on the critique.
4. **Train**: Use the (original, revised) pairs as preference data for DPO/RLHF.

```
Principle: "Choose the response that is most helpful while being harmless."

Original: [Potentially harmful response]
Critique: "This response could be misused because..."
Revision: [Improved, safer response]
```

### Why CAI Matters

- Reduces dependence on human annotators (expensive, slow, inconsistent)
- Scalable — can generate training data automatically
- Principles are explicit and auditable
- Can handle nuanced situations that binary human ratings miss

## When to Use Each Alignment Method

```
Method     | Best For                          | Data Needed               | Complexity
-----------|-----------------------------------|---------------------------|----------
SFT only   | Basic instruction following       | 1K-10K (instruction, response) | Low
DPO        | General preference alignment      | 10K+ preference pairs     | Medium
GRPO       | Verifiable tasks (math, code)     | Prompts + reward function | Medium
PPO/RLHF   | Maximum control, complex rewards  | Reward model + prompts    | High
CAI        | Safety, scalable alignment        | Principles + few examples | High
```

### Decision Flow

```
Do you need the model to follow instructions?
|-- Not yet --> SFT first, always
|-- Already does (SFT done) -->
    |-- Need better quality/safety on subjective tasks?
    |   |-- Have human preference data? --> DPO
    |   |-- Can generate principles? --> CAI then DPO
    |-- Need better reasoning/correctness?
    |   |-- Can verify answers automatically? --> GRPO
    |-- Need maximum control with custom reward?
        |-- PPO (but consider if DPO/GRPO can approximate it first)
```

---

### Check Your Understanding

1. In the alignment method comparison table, SFT requires 1K-10K instruction-response pairs while DPO requires 10K+ preference pairs. Why does DPO typically need more data?
2. You have a task where correct answers can be verified programmatically (e.g., math problems with known solutions). Which alignment method is most appropriate and why?
3. What is Constitutional AI's key advantage over standard RLHF in terms of scalability?

<details>
<summary>Answers</summary>

1. DPO is refining already-learned behavior rather than teaching it from scratch, so the signal per example is weaker. Each preference pair provides a relative signal (A is better than B) rather than an absolute target. The model needs many such comparisons to learn consistent preferences. Additionally, preference data needs to cover diverse failure modes the model might exhibit.
2. GRPO. When you can verify correctness automatically, you have a reliable reward signal without needing human annotations or a learned reward model. GRPO generates groups of responses, scores them with the verifiable reward function, and optimizes the policy based on within-group relative performance. This is exactly how DeepSeek-R1 achieved strong reasoning capabilities.
3. CAI generates its own training data by having the model critique and revise its own responses against explicit principles. This reduces dependence on expensive human annotators, scales automatically, and makes the alignment criteria explicit and auditable rather than implicit in human preference labels.

</details>

---

## Safety and Red Teaming

Alignment doesn't make models perfectly safe. It makes them safer. Understanding the remaining failure modes is critical.

### What Aligned Models Still Get Wrong

1. **Jailbreaks**: Carefully crafted prompts can bypass safety training. "Do Anything Now" (DAN), multi-language attacks, encoding attacks.
2. **Sycophancy**: Models agree with users even when users are wrong, especially on subjective topics.
3. **Refusal over-correction**: Models refuse benign requests because they pattern-match to harmful ones. "How do I kill a Python process?" triggers refusal.
4. **Hallucination**: Alignment reduces but doesn't eliminate confident fabrication.
5. **Reward hacking in the wild**: Models learn to produce outputs that *look* helpful (formatting, structure, confidence) without actually being correct.

### Red Teaming

Red teaming is systematic adversarial testing of aligned models:

- **Human red teams**: Domain experts try to elicit harmful outputs
- **Automated red teaming**: Use one LLM to generate adversarial prompts for another
- **Category-based testing**: Test specific failure modes — bias, toxicity, privacy leaks, dangerous instructions, misinformation
- **Continuous process**: Not a one-time thing. New attacks emerge constantly.

For an e-commerce context: red team your shopping assistant for recommending counterfeit products, leaking other customers' data, providing manipulative sales tactics, or producing discriminatory recommendations.

## The Role of Human Annotators

The entire alignment pipeline depends on human judgment. The quality of your annotators determines the quality of your aligned model.

### What Makes Good Annotation

- **Clear guidelines**: Annotators need specific rubrics, not vague instructions
- **Calibration**: Regular alignment sessions where annotators discuss edge cases
- **Diverse annotators**: Avoid encoding a single demographic's preferences
- **Quality control**: Agreement metrics, gold-standard examples, regular audits

### The Annotation Bottleneck

- Expert annotators are expensive ($30-100/hour for specialized domains)
- Inter-annotator agreement is often surprisingly low (60-80%) on subjective tasks
- Annotator biases get encoded into the model
- Scale is limited — you can't annotate millions of examples

This is why synthetic data generation (using strong models to create preference data) and Constitutional AI (using principles instead of per-example labels) are increasingly important.

## Putting It All Together: Alignment in Practice

A realistic alignment pipeline for a production model:

```
1. Start with a pretrained base model (Llama 3.1 8B)
2. SFT on 50K high-quality instruction-response pairs
   - Mix of general + domain-specific (shopping) conversations
   - 3 epochs, lr=2e-5, cosine schedule
3. Generate preference data:
   - Run the SFT model on 20K prompts, generating 4 responses each
   - Have GPT-4 rank responses (synthetic preferences) + human validation on 2K
4. DPO training on preference data
   - beta=0.1, lr=5e-7, 1-2 epochs
5. Red team the aligned model
   - Fix issues with targeted SFT on failure cases
6. Deploy with guardrails (input/output filters, content moderation)
```

Total cost for step 2-5: $1K-$10K in compute. A fraction of pretraining. This is where applied ML engineers spend most of their time.

## Common Pitfalls

1. **Skipping SFT and going straight to preference optimization.** SFT is not optional. The model must first learn to follow instructions before you can meaningfully optimize preferences. Without SFT, the model's responses are too incoherent for preference pairs to provide a useful learning signal.
2. **Using a low-quality reward model and trusting it blindly.** The reward model is the bottleneck of the RLHF pipeline. If it has verbosity bias, sycophancy, or format hacking tendencies, the policy will optimize for those artifacts rather than genuine quality. Always validate your reward model against held-out human judgments.
3. **Setting DPO beta too low.** A common mistake is using an aggressive beta (e.g., 0.01) hoping for faster alignment. Low beta allows the model to deviate far from the reference, leading to degenerate outputs, mode collapse, or loss of general capabilities. Start with beta=0.1 and adjust carefully.
4. **Neglecting red teaming after alignment.** Alignment does not make models perfectly safe. Jailbreaks, sycophancy, refusal over-correction, and hallucination persist. Red teaming is a continuous process, not a one-time check.

## Hands-On Exercises

### Exercise 1: Construct DPO Preference Pairs (20 min)

Create a small preference dataset for a shopping assistant by generating responses and ranking them.

```python
# Use any available LLM API or a local model to generate responses.
# For each of these 5 prompts, generate 2-3 responses and rank them:
prompts = [
    "What laptop do you recommend for a college student on a budget?",
    "Compare Nike and Adidas running shoes.",
    "I bought a defective product. What should I do?",
    "What are the best gifts under $50?",
    "Is this product worth the price? [link to a $200 headphone]",
]

# For each prompt, create a DPO-format entry:
# {"prompt": "...", "chosen": "best response", "rejected": "worst response"}
# Write 2-3 sentences explaining WHY the chosen response is better.
# Consider: helpfulness, accuracy, tone, specificity, and appropriate length.
```

### Exercise 2: Explore the DPO Loss Function (15 min)

Implement the DPO loss computation manually to build intuition.

```python
import torch
import torch.nn.functional as F

# Simulated log-probabilities for a batch of 4 preference pairs
# These represent log P(response | prompt) under the policy and reference models
policy_log_prob_chosen = torch.tensor([-2.0, -1.5, -3.0, -2.5])
policy_log_prob_rejected = torch.tensor([-2.5, -2.0, -2.8, -2.0])
ref_log_prob_chosen = torch.tensor([-2.1, -1.6, -3.1, -2.6])
ref_log_prob_rejected = torch.tensor([-2.4, -1.9, -2.9, -2.1])

beta = 0.1

# Compute DPO loss:
# L = -log(sigmoid(beta * ((log_pi_chosen - log_ref_chosen) - (log_pi_rejected - log_ref_rejected))))
# Experiment: what happens when you change beta to 0.01 vs 1.0?
# Question: What does a negative chosen-rejected margin mean for the gradient?
```

## Interview Questions

**Conceptual:**
1. Explain the three-stage alignment pipeline (Pretrain -> SFT -> RLHF/DPO). What does each stage accomplish?
2. What is a reward model? How is it trained? What can go wrong?
3. Explain DPO in simple terms. Why is it preferred over PPO in many settings?
4. What is GRPO and when would you use it over DPO?
5. What is Constitutional AI? How does it reduce the need for human annotators?

**Applied:**
6. You're aligning a shopping assistant. It tends to be overly cautious and refuses to recommend products (says "I'm not qualified to recommend products"). How do you fix this?
7. Your aligned model is sycophantic — it always agrees with the customer, even when they describe a product incorrectly. How would you address this?
8. Design a preference data collection pipeline for a customer service chatbot. What do you ask annotators to evaluate? What rubric do you use?
9. You have a model that excels at helpfulness but sometimes recommends competitor products. You can easily verify whether a response mentions competitors. Which alignment method would you use and why?
10. After DPO training, your model's general conversation quality degraded even though shopping-specific metrics improved. Diagnose and fix.

**Answer to Q9**: GRPO would be ideal here. You have an easily verifiable reward signal — does the response mention a competitor? (yes=0, no=1, combined with a helpfulness score). Generate groups of responses, score them automatically, and use GRPO to optimize. No human preference pairs needed for this specific dimension. You could combine with DPO for the subjective quality dimensions.

## Summary

This lesson covered how models are aligned with human preferences after pretraining and SFT. Key takeaways:

- **The alignment pipeline** has three stages: pretraining (raw intelligence), SFT (instruction following), and preference optimization (quality and safety). Each is necessary.
- **Reward models** learn to score responses based on human preferences using pairwise ranking loss, but they are the bottleneck -- verbosity bias, sycophancy, and format hacking are common failure modes.
- **PPO/RLHF** is powerful but complex, requiring four models in memory and careful hyperparameter tuning. The KL divergence penalty prevents reward hacking but adds a critical tuning knob (beta).
- **DPO** simplifies preference optimization to a supervised loss function, eliminating the reward model and value model. It requires only preference pairs and a reference model.
- **GRPO** excels at tasks with verifiable rewards by using group-relative advantages, eliminating both the reward model and value model. It was key to DeepSeek-R1's reasoning capabilities.
- **Constitutional AI** uses self-critique against explicit principles to generate training data, reducing reliance on human annotators.
- **Alignment is not safety.** Red teaming, guardrails, and continuous monitoring are still required after alignment.

## What's Next

The next lesson, **Inference Optimization** (see [Inference Optimization](../inference-optimization/COURSE.md)), covers how to serve aligned models efficiently at scale, including KV caching, quantization, speculative decoding, and cost optimization strategies. Training happens once; inference happens millions of times -- optimization here has the highest ROI.
