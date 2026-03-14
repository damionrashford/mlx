## Learning Objectives

By the end of this lesson, you will:

**Understand:**
- How self-attention computes weighted combinations of all input positions using Q, K, V projections, why scaling by sqrt(d_k) prevents softmax saturation, and how multi-head attention learns multiple relationship types in parallel
- How positional encodings (sinusoidal, learned, RoPE, ALiBi) inject sequence order into the permutation-invariant attention mechanism, and why RoPE's relative position encoding via rotation is the modern standard
- How the complete transformer block (RMSNorm, multi-head attention, residual connection, SwiGLU FFN) processes information, and the distinct roles of attention (routing between positions) and FFN (processing within positions)

**Apply:**
- Trace the full attention computation with concrete tensor shapes: from input X through Q, K, V projections, scaled dot-product scores, causal masking, softmax, and value-weighted output
- Select the appropriate transformer variant (encoder-only, decoder-only, encoder-decoder) for a given task, and configure KV cache, GQA, and Flash Attention for efficient inference

**Analyze:**
- Evaluate the tradeoffs between context length, KV cache memory, compute cost, and attention quality for production deployment of transformer models, considering techniques like GQA, Flash Attention, and RoPE scaling

## Prerequisites

- **Layers, activations, and residual connections** -- transformers build on feedforward layers, non-linear activations, and skip connections covered in the fundamentals (see [Fundamentals](../fundamentals/COURSE.md))
- **Dot products and matrix multiplication** -- the attention computation is fundamentally a series of dot products and matrix multiplications, and understanding tensor shapes is essential for following the math (see [Linear Algebra](../../01-math-foundations/linear-algebra/COURSE.md))
- **Normalization techniques** -- transformers use LayerNorm/RMSNorm at every block, and understanding why normalization stabilizes training is important (see [Training Mechanics](../training-mechanics/COURSE.md))

---

# Transformers and Attention: The Architecture That Changed Everything

This is the most important document in this course. Every frontier AI system — GPT-4, Claude, Gemini, Llama, Stable Diffusion, DALL-E, Whisper — is a transformer. If you understand transformers deeply, you understand modern AI.

---

## 1. Self-Attention From First Principles

### The Problem Attention Solves

Consider processing the sentence: "The animal didn't cross the street because **it** was too tired."

What does "it" refer to? The animal. A human knows this instantly because we consider the meaning of every word in relation to every other word. We do not process left-to-right with a lossy memory — we hold the entire sentence in mind and reason about relationships.

Self-attention gives neural networks this ability. Every position in the sequence can directly query every other position to determine relevance. No information bottleneck, no sequential processing, no lossy compression.

### The Fundamental Operation

Self-attention takes a sequence of vectors and outputs a new sequence of vectors, where each output is a **weighted combination of all input vectors**. The weights are determined by how "relevant" each input is to the position being computed.

```
Input:  [v1, v2, v3, v4, v5]   (5 vectors, one per token)
Output: [o1, o2, o3, o4, o5]   (5 new vectors)

o3 = 0.05*v1 + 0.10*v2 + 0.50*v3 + 0.30*v4 + 0.05*v5
     ^--- attention weights (sum to 1, from softmax)
```

The output for position 3 is a mixture of all positions, weighted by relevance. If token 4 is very relevant to token 3, it gets a high weight. This relevance is not hardcoded — it is **learned**.

### Why This Is Revolutionary

Compare to previous approaches:
- **Feedforward networks:** Each position is processed independently. No cross-position information.
- **RNNs:** Position t gets information from 1 through t-1, but mediated through a lossy hidden state. Information from distant positions is degraded.
- **Self-attention:** Every position directly accesses every other position with O(1) path length. No degradation, no bottleneck.

### Why Not Just Use the Raw Vectors?

Because a token needs to play three different roles simultaneously:
1. **"What am I looking for?"** (when I am the one querying)
2. **"What do I advertise about myself?"** (when others are querying me for compatibility)
3. **"What information do I provide?"** (when I am selected as relevant and contribute to the output)

These three roles require different representations of the same token. This is why we need Query, Key, and Value projections.

---

## 2. Query, Key, Value — The Heart of Attention

### The Search Engine Analogy

This is the clearest way to understand Q, K, V:

- **Query (Q):** Your search query. "What am I looking for?" When you type "best restaurants in Toronto," the query encodes your intent.
- **Key (K):** The title/metadata of each document in the index. "What is this document about?" Each document advertises its content through its key.
- **Value (V):** The actual content of each document. "What information does this contain?" When a document is matched, you get its value (the actual page content).

The search engine compares your query against all keys to find the best matches, then returns the corresponding values weighted by match quality.

### The Math

Each input vector `x_i` is projected into three different spaces by learned weight matrices:

```python
Q = X @ W_Q    # (seq_len, d_model) @ (d_model, d_k) -> (seq_len, d_k)
K = X @ W_K    # (seq_len, d_model) @ (d_model, d_k) -> (seq_len, d_k)
V = X @ W_V    # (seq_len, d_model) @ (d_model, d_v) -> (seq_len, d_v)
```

`W_Q`, `W_K`, `W_V` are learned parameters. They project the same input into three different representations optimized for the three different roles.

### Why Separate Projections Matter — A Deep Example

Consider the word "bank" in two sentences:
- "I went to the **bank** to deposit money." (financial institution)
- "I sat on the river **bank**." (riverside)

The same token "bank" needs to:
- Generate different **queries** (looking for financial vs geographical context)
- Generate different **keys** (advertising "finance" vs "nature")
- Generate different **values** (providing financial vs geographical information)

The learned projection matrices enable this. After the first few layers of a transformer process the surrounding context, the embedding for "bank" already carries contextual information. The Q, K, V projections then extract the appropriate signals for the attention mechanism.

### Another Intuition: Matchmaking

Think of Q and K as two sides of a matchmaking system:
- Q says: "I need a subject noun to agree with my verb"
- K says: "I am a singular noun"
- If Q and K match well (high dot product), V's information gets passed along

The dot product between Q and K is a **compatibility score** — how well do these two tokens relate? High compatibility means the value should be weighted heavily in the output.

---

## 3. The Attention Computation: Step by Step

### The Full Formula

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

Let us walk through every single step with concrete shapes and numbers.

### Step 1: Compute Compatibility Scores (Q @ K^T)

```python
scores = Q @ K.T    # (seq_len, d_k) @ (d_k, seq_len) -> (seq_len, seq_len)
```

This produces a **square matrix** where entry (i, j) measures how much position i should attend to position j. The dot product of query i with key j is the raw compatibility score.

For a 5-token sequence with d_k = 4:

```
Q (5x4):         K^T (4x5):        scores (5x5):
[q1]              [k1 k2 k3 k4 k5]   [q1.k1  q1.k2  q1.k3  q1.k4  q1.k5]
[q2]       @      [.. .. .. .. ..]  = [q2.k1  q2.k2  q2.k3  q2.k4  q2.k5]
[q3]              [.. .. .. .. ..]    [q3.k1  q3.k2  q3.k3  q3.k4  q3.k5]
[q4]              [.. .. .. .. ..]    [q4.k1  q4.k2  q4.k3  q4.k4  q4.k5]
[q5]                                  [q5.k1  q5.k2  q5.k3  q5.k4  q5.k5]
```

Each row of the scores matrix represents one token's compatibility with every other token. Row 3 tells us how much token 3 should attend to each of the 5 tokens.

### Step 2: Scale by sqrt(d_k)

```python
scores = scores / sqrt(d_k)    # Prevent softmax saturation
```

This is critical. Without scaling, the dot products grow in magnitude with d_k, causing softmax to produce near-one-hot distributions where one token gets all the attention and gradients die. See Section 4 for the full explanation.

### Step 3: Apply Softmax (Row-wise)

```python
weights = softmax(scores, dim=-1)    # Each row sums to 1
```

After softmax, each row is a probability distribution. Row 3 might look like:

```
[0.05, 0.10, 0.50, 0.30, 0.05]
 ^                              ^
 Token 1 barely relevant    Token 5 barely relevant
              ^      ^
         Self-attention  Token 4 quite relevant
```

### Step 4: Weighted Combination of Values (weights @ V)

```python
output = weights @ V    # (seq_len, seq_len) @ (seq_len, d_v) -> (seq_len, d_v)
```

The output for each position is a weighted sum of all value vectors:

```
o3 = 0.05*v1 + 0.10*v2 + 0.50*v3 + 0.30*v4 + 0.05*v5
```

Position 3's output is dominated by its own value (weight 0.50) and position 4's value (weight 0.30), with small contributions from other positions.

### What the Attention Matrix Looks Like

For a 5-token sentence "The cat sat on mat":

```
         The    cat    sat    on     mat
The    [ 0.50   0.20   0.10   0.10   0.10 ]
cat    [ 0.15   0.40   0.25   0.10   0.10 ]
sat    [ 0.10   0.30   0.30   0.20   0.10 ]
on     [ 0.05   0.10   0.20   0.40   0.25 ]
mat    [ 0.10   0.10   0.15   0.25   0.40 ]
```

"cat" attends strongly to itself and to "sat" (what the cat did). "on" attends to itself and "mat" (what it is on). These patterns are learned — the network discovers which relationships matter for the task.

### Complexity Analysis

- **Time complexity:** O(T^2 * d) where T is sequence length and d is dimension
- **Memory complexity:** O(T^2) for the attention matrix
- This quadratic scaling is the main limitation — it is why context windows exist and why Flash Attention was invented

---

## 4. Why Divide by sqrt(d_k): Preventing Softmax Saturation

### The Problem Without Scaling

The dot product `q . k` grows in magnitude with the dimension `d_k`. If `q` and `k` are random vectors with unit variance in each component:

```
E[q . k] = 0           (expected value is zero)
Var[q . k] = d_k        (variance grows linearly with dimension)
Std[q . k] = sqrt(d_k)  (standard deviation)
```

For `d_k = 512`, the dot products have standard deviation `sqrt(512) = 22.6`. Values like +45 or -30 are common. When these go into softmax:

```
softmax([45, 30, 2, -10, -20]) = [~1.0, ~0.0, ~0.0, ~0.0, ~0.0]
```

The softmax saturates — one position gets essentially all the attention. This causes two problems:
1. **Gradient vanishing:** The gradient of softmax at saturation is nearly zero. The model cannot learn to shift attention to other positions.
2. **Loss of nuance:** The model cannot attend to multiple relevant positions simultaneously. Attention becomes a hard lookup instead of a soft mixture.

### The Fix

Dividing by `sqrt(d_k)` normalizes the variance of the dot products back to 1, regardless of dimension:

```
Var[q . k / sqrt(d_k)] = Var[q . k] / d_k = d_k / d_k = 1
```

Now softmax receives inputs with unit variance:

```
softmax([45/22.6, 30/22.6, 2/22.6, -10/22.6, -20/22.6])
= softmax([2.0, 1.3, 0.09, -0.44, -0.88])
= [0.46, 0.23, 0.07, 0.04, 0.03]
```

A smooth distribution where the model can attend to multiple positions and where gradients flow to all of them.

### Why sqrt(d_k) Specifically?

The variance of a dot product of two d-dimensional vectors with unit-variance components is exactly d. The standard deviation is sqrt(d). Dividing by the standard deviation normalizes to unit variance. This is the statistically correct scaling factor.

### Temperature Scaling — The General Case

The scaling factor `1/sqrt(d_k)` is actually a special case of **temperature scaling** in softmax:

```
softmax(z / T)
```

- High temperature (T > 1): Flatter distribution, more uniform attention
- Low temperature (T < 1): Sharper distribution, more focused attention
- T = sqrt(d_k): The default that preserves unit variance

Some architectures learn the temperature as a parameter. In CLIP's contrastive learning, the temperature is learned and is crucial for performance.

---

### Check Your Understanding

1. Walk through the attention computation for a single head with seq_len=3 and d_k=4. If Q and K are both 3x4 matrices, what is the shape of the scores matrix, and what does each entry represent? After softmax and multiplication by V (3x4), what is the output shape?
2. Why must each token have separate Q, K, and V projections? What would go wrong if we used the same projection for all three (i.e., Q = K = V = X @ W)?
3. Without the 1/sqrt(d_k) scaling, what happens to the attention distribution as d_k increases from 64 to 512? How does this affect gradient flow through the attention layer?

<details>
<summary>Answers</summary>

1. Scores = Q @ K^T has shape (3, 3), where entry (i, j) is the dot product of query i and key j -- the raw compatibility score measuring how much position i should attend to position j. After scaling and softmax (applied row-wise), each row becomes a probability distribution summing to 1. Output = weights @ V has shape (3, 4), where each row is a weighted combination of all value vectors for that position.
2. If Q = K = V = X @ W, every token would query, advertise, and provide the same representation. The attention score between positions i and j would be symmetric (q_i . k_j = q_j . k_i), meaning token A's interest in token B equals token B's interest in token A. This eliminates the ability to learn asymmetric relationships (e.g., "the" should attend strongly to its noun, but the noun does not need to attend as strongly to "the"). Separate projections allow each role to be optimized independently.
3. As d_k increases, the variance of dot products grows linearly (Var = d_k), so standard deviations grow as sqrt(d_k). For d_k=512, dot products have std ~22.6, producing values like +45 and -30. Softmax of such large values saturates to a near-one-hot distribution where one position gets ~100% of attention. The gradient of softmax at saturation is near zero, so the model cannot learn to redistribute attention -- gradient flow through the attention layer is effectively blocked.

</details>

---

## 5. Multi-Head Attention

### Why Multiple Heads?

A single attention head learns **one type of relationship**. But language (and data in general) has many simultaneous relationship types:

- **Syntactic:** subject-verb agreement ("The cats **are**")
- **Semantic:** word meaning associations ("bank" -> "money" or "river")
- **Positional:** what is nearby in the sequence
- **Coreference:** pronoun resolution ("it" -> "cat")
- **Logical:** causal relationships ("because" -> connecting cause and effect)
- **Idiomatic:** phrase-level patterns ("kick the bucket" -> death)

One attention head would have to multiplex all of these into a single set of attention weights. Multi-head attention runs multiple attention operations in parallel, each with its own Q, K, V projections — each head can specialize in a different relationship type.

### The Computation

```python
# Instead of one attention with d_model dimensions:
# Run h heads, each with d_k = d_model / h dimensions

head_1 = Attention(X @ W_Q1, X @ W_K1, X @ W_V1)   # (seq_len, d_k)
head_2 = Attention(X @ W_Q2, X @ W_K2, X @ W_V2)   # (seq_len, d_k)
...
head_h = Attention(X @ W_Qh, X @ W_Kh, X @ W_Vh)   # (seq_len, d_k)

# Concatenate all heads:
multi_head = Concat(head_1, ..., head_h)              # (seq_len, h * d_k = d_model)

# Final projection to mix head outputs:
output = multi_head @ W_O                              # (seq_len, d_model)
```

### The Dimensionality Trick

With `d_model = 512` and `h = 8` heads, each head operates in `d_k = 64` dimensions. The total computation is approximately the same as a single head with 512 dimensions (same number of total parameters), but the model can learn 8 **independent** relationship types.

The final `W_O` projection mixes information across heads, allowing the model to combine insights from different relationship types into a unified representation.

### What Different Heads Actually Learn

Researchers have probed transformer heads and found distinct specializations:

| Head Type | What It Learns | Example |
|-----------|---------------|---------|
| **Positional heads** | Attend to the previous or next token | Bigram statistics |
| **Syntactic heads** | Subject-verb, modifier-noun relationships | "cats ... are" |
| **Semantic heads** | Attend to semantically similar words | "doctor" -> "patient" |
| **Induction heads** | Detect and reproduce patterns | If [A][B] appeared before, when [A] appears again, attend to [B] |
| **Copy heads** | Attend to tokens that should be copied to output | Names, numbers |
| **Rare token heads** | Focus on unusual or important tokens | Keywords, entities |

This specialization emerges **entirely from training** — nobody programs these patterns. The loss function creates pressure to capture all types of relationships, and multi-head attention provides the capacity for specialization.

### Induction Heads — A Key Discovery

Induction heads (Olsson et al., 2022) are particularly important. They implement a two-step pattern:
1. Head A (in an earlier layer) copies the token after the current token's previous occurrence
2. Head B (in a later layer) uses that information to predict what comes next

This mechanism is believed to be responsible for **in-context learning** — the ability of LLMs to learn from examples in the prompt without weight updates. Understanding induction heads is understanding a fundamental mechanism of how transformers work.

### Typical Head Counts

| Model | Heads | d_model | d_k per head |
|-------|-------|---------|-------------|
| BERT-base | 12 | 768 | 64 |
| GPT-2 | 12 | 768 | 64 |
| GPT-3 | 96 | 12288 | 128 |
| Llama-3-8B | 32 | 4096 | 128 |
| Llama-3-70B | 64 | 8192 | 128 |

---

## 6. Positional Encoding

### Why Position Information Is Needed

Self-attention is **permutation invariant** — it treats the input as a set, not a sequence. "The cat sat on the mat" and "mat the on sat cat the" would produce identical attention scores without positional encoding. We must inject position information explicitly.

This is fundamentally different from RNNs and CNNs, which have position information built into their architecture (sequential processing and local receptive fields, respectively).

### Sinusoidal Positional Encoding (Original Transformer, 2017)

```python
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Each dimension oscillates at a different frequency. Low-frequency dimensions encode coarse position (beginning vs end). High-frequency dimensions encode fine position (adjacent tokens).

**Key properties:**
- Deterministic (no learned parameters)
- Can generalize to any sequence length (the functions are defined for all positions)
- The encoding of position `pos + k` can be expressed as a linear function of the encoding at `pos`, which lets the model learn relative position patterns through linear attention
- Added to (not concatenated with) the token embeddings

### Learned Positional Embeddings (BERT, GPT-2)

Just learn a separate embedding vector for each position:

```python
self.pos_embedding = nn.Embedding(max_seq_len, d_model)  # e.g., 512 x 768
x = token_embedding + self.pos_embedding(position_ids)
```

Simple and effective, but has a fixed maximum length. Cannot generalize to sequences longer than `max_seq_len`. BERT's 512-token limit was partly due to this design.

### Rotary Positional Embeddings (RoPE) — The Modern Standard

RoPE (Su et al., 2021) is used in virtually all modern LLMs: Llama, Mistral, Qwen, Gemma, and many others.

**Core idea:** Encode position by **rotating** the query and key vectors in 2D subspaces. The rotation angle is proportional to the position.

```python
# For each pair of dimensions (2i, 2i+1), at position pos:
theta_i = pos * base_freq_i        # angle depends on position and frequency
q_rotated[2i]   = q[2i]*cos(theta) - q[2i+1]*sin(theta)
q_rotated[2i+1] = q[2i]*sin(theta) + q[2i+1]*cos(theta)
# Same rotation applied to keys
```

**Why RoPE is brilliant:** When you compute the dot product `q_i . k_j` (both rotated), the result depends on `(pos_i - pos_j)` — the **relative** distance between the two tokens. The absolute positions cancel out through the rotation math.

This means:
- The model naturally learns **relative position patterns** ("the word 3 positions before me") without explicit relative position matrices
- The dot product between adjacent tokens is the same regardless of their absolute position in the sequence
- The representation is local: tokens can distinguish their neighbors from distant tokens

**Extrapolation:** RoPE can extend to sequences longer than seen during training (with some quality degradation). Techniques like:
- **NTK-aware scaling:** Multiply the base frequency by a factor, spreading out the rotation angles
- **YaRN (Yet Another RoPE Extension):** Scale different frequency bands differently
- **Dynamic NTK:** Adjust scaling based on the actual sequence length at inference time

These techniques are how models trained on 4K tokens can handle 128K+ tokens at inference.

### ALiBi (Attention with Linear Biases)

ALiBi (Press et al., 2022) takes a completely different approach — no modification to embeddings at all:

```python
# Add a distance-based penalty directly to attention scores:
attention_score(i, j) = q_i . k_j - m * |i - j|
# m is a head-specific constant (different heads, different slopes)
```

Closer tokens get higher attention scores (smaller penalty). Each head has a different slope `m`, so some heads prefer very local context while others maintain broader attention.

**Advantages:**
- No learned parameters for position
- Extrapolates naturally to longer sequences (the penalty function generalizes)
- Trivially simple to implement
- The penalty is interpretable: tokens k positions away are penalized by m*k

**Disadvantage:** The linear bias is a relatively crude position signal. RoPE captures richer positional interactions.

### Comparison Table

| Method | Type | Extrapolation | Parameters | Used In |
|--------|------|--------------|------------|---------|
| Sinusoidal | Absolute, fixed | Moderate | 0 | Original Transformer |
| Learned | Absolute, learned | None (hard limit) | max_len * d_model | BERT, GPT-2 |
| RoPE | Relative, rotation | Good (with scaling) | 0 | Llama, Mistral, Qwen |
| ALiBi | Relative, bias | Excellent | 0 | BLOOM, MPT |

---

## 7. The Feed-Forward Network (FFN)

### What It Does

After attention, each position passes through a two-layer MLP **independently** (no cross-position interaction):

```python
# Standard FFN:
FFN(x) = GELU(x @ W1 + b1) @ W2 + b2
# W1: (d_model, d_ff)    -- expand to wider dimension
# W2: (d_ff, d_model)     -- compress back
# d_ff is typically 4 * d_model (e.g., 4096 -> 16384)
```

### The Two-Phase View of Transformer Processing

Each transformer layer does two distinct things:
1. **Attention:** Mix information **between** positions. "Who should I listen to?"
2. **FFN:** Transform information **within** each position. "Given what I heard, what should I compute?"

Attention is the router. The FFN is the processor. Both are essential.

### What the FFN Actually Computes

Research (Geva et al., 2021) suggests the FFN acts as a **key-value memory**:

```
First layer (W1):  "key" patterns that match certain input contexts
Activation (GELU): Gate that selects matching patterns
Second layer (W2): "value" information to inject for each matched pattern
```

Each neuron in the first layer is a "key" that detects certain input patterns. When it activates (GELU output is high), the corresponding column of W2 is the "value" — the information to inject into the representation.

**Example:** A neuron might activate when the context suggests a location is being discussed, and its W2 column pushes the representation toward geography-related features. Another neuron might activate for temporal expressions and push toward time-related features.

This means the FFN stores **factual knowledge** about the world. When you fine-tune a model and change factual knowledge, you are primarily modifying the FFN weights.

### GLU Variants — The Modern Standard

Most modern transformers (Llama, Mistral, Gemma, Qwen) use **Gated Linear Units** instead of a standard FFN:

```python
# SwiGLU (used in Llama, Mistral):
def swiglu(x, W_gate, W_up, W_down):
    gate = silu(x @ W_gate)     # Gate signal (what to pass)
    up = x @ W_up               # Value signal (what to pass through)
    return (gate * up) @ W_down  # Element-wise gating, then compress

# Three weight matrices instead of two
# But better performance per parameter
# d_ff is typically 8/3 * d_model (to match parameter count of standard FFN)
```

The gating mechanism lets the network selectively pass or block information dimension by dimension, providing finer control than a simple nonlinearity.

### GeGLU, ReGLU, SwiGLU

These are all variations of the gated FFN that differ only in the activation function used for the gate:

| Variant | Gate Activation | Used In |
|---------|----------------|---------|
| SwiGLU | SiLU (Swish) | Llama, Mistral, Gemma |
| GeGLU | GELU | Some research models |
| ReGLU | ReLU | Less common |

SwiGLU has become the dominant choice empirically.

---

## 8. Layer Normalization

### What It Does

LayerNorm normalizes the activations across the feature dimension for each individual sample and position:

```python
# For each token position independently:
mean = x.mean(dim=-1, keepdim=True)       # Mean across d_model
std = x.std(dim=-1, keepdim=True)         # Std across d_model
x_norm = (x - mean) / (std + eps)         # Normalize to zero mean, unit variance
output = gamma * x_norm + beta            # Learned scale (gamma) and shift (beta)
```

### Why It Helps

Without normalization, activations can drift to extreme values as they pass through layers:
- Activations grow larger -> softmax saturates -> gradients vanish
- Activations drift -> loss landscape becomes rugged -> training is unstable
- Different layers operate at different scales -> learning rates cannot be optimal for all layers

LayerNorm keeps activations centered and scaled at every layer, making the loss landscape smoother and enabling higher learning rates and faster convergence.

### LayerNorm vs BatchNorm

| Property | LayerNorm | BatchNorm |
|----------|-----------|-----------|
| Normalizes across | Feature dimension (d_model) | Batch dimension |
| Depends on batch size? | No | Yes |
| Works for variable-length sequences? | Yes | Poorly |
| Works with batch size 1? | Yes | No (no batch statistics) |
| Used in | Transformers | CNNs |
| During inference | Same as training | Uses running mean/var |

Transformers use LayerNorm because (1) sequences have variable lengths, making batch statistics ill-defined, and (2) it is independent of batch size, which matters for large model training where batch sizes can be very small per GPU.

### Pre-Norm vs Post-Norm — A Critical Architectural Choice

**Post-norm** (original transformer paper, 2017):
```python
x = LayerNorm(x + Attention(x))     # Norm AFTER the residual
x = LayerNorm(x + FFN(x))
```

**Pre-norm** (GPT-2, all modern LLMs):
```python
x = x + Attention(LayerNorm(x))     # Norm BEFORE the sub-layer
x = x + FFN(LayerNorm(x))
```

**Why pre-norm won:**
1. **The residual connection directly connects input to output** without any normalization in between. Gradients flow through the identity path completely unmodified.
2. Training is **much more stable** for deep models (50+ layers). Post-norm requires careful learning rate warmup; pre-norm is more forgiving.
3. Pre-norm transformers can be trained **without learning rate warmup** (though warmup still helps in practice).
4. The gradient norm stays bounded regardless of depth (provably, for pre-norm).

**Disadvantage of pre-norm:** Some research suggests post-norm achieves slightly better final performance with careful hyperparameter tuning. But the stability advantage of pre-norm far outweighs this in practice.

### RMSNorm — The Simplified Standard

RMSNorm (Zhang and Sennrich, 2019) drops the mean-centering step, using only root-mean-square normalization:

```python
rms = sqrt(mean(x^2) + eps)
output = (x / rms) * gamma
# No beta (shift) parameter, no mean subtraction
```

**Why it works as well as LayerNorm:** The re-centering (subtracting the mean) provides minimal benefit empirically but costs extra computation. RMSNorm achieves the same training dynamics with ~15% less computation.

Used in: Llama, Mistral, Gemma, and virtually all modern LLMs.

---

### Check Your Understanding

1. A transformer has d_model=512 and 8 attention heads. What is d_k per head? How does the total computational cost of multi-head attention compare to single-head attention with d_k=512?
2. Why do modern LLMs use pre-norm (LayerNorm before the sub-layer) instead of post-norm (LayerNorm after the residual)? What specific property of the gradient changes?
3. The FFN in a transformer is applied independently to each position. If it processes tokens independently, why is it useful -- is not all the cross-position work done by attention?

<details>
<summary>Answers</summary>

1. d_k = d_model / h = 512 / 8 = 64 per head. The total computational cost is approximately the same: 8 heads each doing attention in 64 dimensions involves the same number of total multiply-adds as one head in 512 dimensions (since the QKV projections produce the same total number of parameters). The key advantage is not efficiency but capacity: 8 heads can learn 8 independent relationship types.
2. With pre-norm, the residual connection directly connects input to output without any normalization in between: x_out = x + Attention(Norm(x)). This means the gradient from the loss flows through the identity path completely unmodified (gradient of addition is 1). With post-norm, the gradient must also pass through the normalization layer, which can distort it. Pre-norm guarantees bounded gradient norm regardless of depth, making training stable for 50+ layer models without requiring careful warmup.
3. Attention determines what information each position should gather from other positions, but it does not do the heavy computation on that gathered information. The FFN acts as a position-wise processor: it takes the attention-enriched representation and transforms it through a wider hidden layer (4x expansion). Research shows the FFN acts as a key-value memory storing factual knowledge -- specific neurons activate for specific contexts and inject corresponding information. Attention is the router; the FFN is where the computation and knowledge retrieval happen.

</details>

---

## 9. Encoder vs Decoder: When to Use Each

### Encoder-Only (BERT Architecture)

Processes the full input with **bidirectional attention** — every token can attend to every other token, including those that come after it.

```
Input:  [CLS] The cat sat on the mat [SEP]
        Each token attends to ALL other tokens (full attention matrix)
Output: Contextual embedding for each token
        [CLS] embedding used for classification
```

**Use cases:**
- Text classification (sentiment, topic, intent)
- Named entity recognition
- Sentence/document embeddings for retrieval
- Semantic similarity
- Extractive question answering (find the answer span in the context)

**Examples:** BERT, RoBERTa, DeBERTa, E5, sentence-transformers

**Key feature:** The [CLS] token's representation (or pooled output) summarizes the entire input. For retrieval, all tokens' representations can be pooled.

**Training objective:** Masked language modeling (MLM). Randomly mask 15% of tokens and predict them from context. This forces bidirectional understanding.

### Decoder-Only (GPT Architecture)

Processes tokens left-to-right with **causal (masked) attention** — each token can only attend to itself and previous tokens. Future tokens are masked.

```
Input:  The cat sat on the
        Token 3 "sat" can only attend to "The", "cat", "sat" (not "on", "the")
Output: Predicted next token at each position
        After "the" -> predict "mat"
```

**Use cases:**
- Text generation, code generation, chat
- General-purpose LLMs (anything framed as "generate the answer")
- In-context learning (few-shot prompting)

**Examples:** GPT-4, Claude, Llama, Mistral, Gemini

**Why decoder-only dominates 2024-2026:** It can do **everything**. Classification? Generate the label. Summarization? Generate the summary. Translation? Generate the translation. Code? Generate the code. The autoregressive objective (predict next token) turns out to be a universal training signal that produces general-purpose intelligence.

**Training objective:** Next token prediction. Given tokens 1 through t, predict token t+1. Simple, scalable, and shockingly effective.

### Encoder-Decoder (T5, BART)

Encoder processes the input bidirectionally. Decoder generates the output autoregressively, attending to both previous output tokens (via causal self-attention) and the full encoder representation (via **cross-attention**).

```
Encoder:  "Translate: The cat sat on the mat" -> bidirectional processing
Cross-attention: Decoder queries attend to encoder keys/values
Decoder:  Generates "Le chat etait assis sur le tapis" one token at a time
```

**Use cases:**
- Translation
- Summarization
- Structured generation where input and output are clearly different
- Tasks where the input should be fully understood before generation begins

**Examples:** T5, BART, Flan-T5, mBART, Whisper (for speech)

**Cross-attention:** Same as self-attention, but Q comes from the decoder and K, V come from the encoder. This lets each decoder position "look at" the entire input while generating.

```python
# Self-attention:  Q, K, V all from the same sequence
# Cross-attention: Q from decoder, K and V from encoder
cross_attn_output = Attention(
    Q = decoder_state @ W_Q,
    K = encoder_output @ W_K,
    V = encoder_output @ W_V
)
```

### The Decision Framework

```
Task requires understanding only (classification, retrieval)?
  -> Encoder-only (BERT, sentence-transformers)

Task requires generation?
  -> Is it a transformation with clear input/output structure (translation)?
       -> Encoder-decoder (T5) or decoder-only (GPT)
     Is it open-ended generation (chat, code)?
       -> Decoder-only (GPT)

Not sure?
  -> Decoder-only. It works for everything, and the ecosystem is largest.
```

---

## 10. Causal Masking

### Preventing Future Token Leakage

In a decoder, position 3 must not attend to positions 4, 5, 6, ... — those tokens have not been generated yet. During training, all tokens are present simultaneously (teacher forcing), so we must **mask** future positions to prevent information leakage.

```python
# Causal mask: lower-triangular matrix
mask = torch.tril(torch.ones(seq_len, seq_len))

# For seq_len = 4:
# [[1, 0, 0, 0],    Token 1 sees only token 1
#  [1, 1, 0, 0],    Token 2 sees tokens 1-2
#  [1, 1, 1, 0],    Token 3 sees tokens 1-3
#  [1, 1, 1, 1]]    Token 4 sees tokens 1-4

# Apply before softmax:
scores = scores.masked_fill(mask == 0, float('-inf'))
weights = softmax(scores)
# -inf becomes 0 after softmax -> future positions get zero attention weight
```

### Why -inf and Not 0?

Setting the raw scores (before softmax) to `-inf` is different from setting the attention weights (after softmax) to 0:

```
# If we zero the weights after softmax:
weights = [0.3, 0.4, 0.3, 0, 0]  # Still sums to 1.0? No! Sums to 1.0 only if we renormalize

# If we set scores to -inf before softmax:
scores = [2.0, 3.0, 1.5, -inf, -inf]
weights = softmax(scores) = [0.24, 0.65, 0.11, 0.0, 0.0]  # Sums to 1.0 automatically
```

Using `-inf` before softmax ensures proper normalization — the remaining attention weights sum to exactly 1.0 without any extra step.

### Why This Enables Autoregressive Generation

During training, we process the entire sequence at once (teacher forcing) but use the causal mask to ensure each position's prediction only uses past context. Each position learns to predict the next token given only what came before.

During inference, we generate one token at a time:
1. Process all tokens so far
2. Take the output at the last position
3. Sample or argmax to get the next token
4. Append it and repeat

The causal mask makes training **equivalent** to inference — the same information is available at each position in both cases. This eliminates the exposure bias problem that plagued RNN-based seq2seq models.

### Prefix LM and Bidirectional Prefixes

Some models (PaLM, UL2) use a **prefix LM** setup: the prompt is processed with bidirectional (unmasked) attention, and only the generation portion uses causal masking. This gives the model full understanding of the prompt while maintaining autoregressive generation.

---

## 11. The Full Transformer Block

### Modern Architecture (Pre-Norm with SwiGLU)

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x, mask=None):
        # Sub-block 1: Attention with residual
        x = x + self.attn(self.norm1(x), mask=mask)

        # Sub-block 2: FFN with residual
        x = x + self.ffn(self.norm2(x))

        return x
```

### What Happens at Each Step — In Detail

1. **RMSNorm** the input: Normalize activations to prevent drift. Each token's d_model-dimensional vector is scaled to have unit RMS.

2. **Multi-head self-attention**: Every token computes Q, K, V projections. Each head independently computes attention scores (Q@K^T/sqrt(d_k)), applies causal mask, softmax, and value-weighted sum. Heads are concatenated and projected through W_O. Result: each token's representation is updated by incorporating information from other tokens.

3. **Add residual**: The input bypasses the attention layer entirely and is added to the output. This creates the gradient highway (gradient of at least 1 through the identity path) and ensures information is not lost.

4. **RMSNorm** again: Re-normalize for the FFN.

5. **SwiGLU FFN**: Each token is processed independently. The gated MLP expands to a wider dimension (8/3 * d_model), applies element-wise gating, and compresses back. This is where **factual knowledge** and **computational processing** happen.

6. **Add residual**: Again, the identity path preserves information.

### The Full Model

```python
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_seq_len):
        self.embed = nn.Embedding(vocab_size, d_model)
        # RoPE is applied inside attention, not as a separate embedding
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.norm_final = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying: self.lm_head.weight = self.embed.weight (common optimization)

    def forward(self, token_ids):
        x = self.embed(token_ids)             # (batch, seq_len, d_model)
        # RoPE applied inside each attention layer
        for block in self.blocks:
            x = block(x, mask=causal_mask)    # N layers of processing
        x = self.norm_final(x)                # Final normalization
        logits = self.lm_head(x)              # (batch, seq_len, vocab_size)
        return logits
```

That is the complete architecture. A GPT-scale model is this same structure repeated 32-80 times with larger dimensions. The architecture is simple — the scale and data are what make it powerful.

### Weight Tying

Many models share weights between the input embedding and the output projection (lm_head). Since both map between d_model and vocab_size dimensions, this halves the embedding parameters (which can be 20-30% of total parameters for smaller models) with minimal quality impact. Almost all modern models use weight tying.

---

## 12. Scaling Laws: Why Bigger Is Better

### The Chinchilla Laws

Hoffmann et al. (2022) established that optimal performance requires scaling **both** model size and data proportionally:

```
Optimal tokens = 20 * parameters

GPT-3 (175B params) was trained on 300B tokens (undertrained by ~20x)
Chinchilla (70B params) trained on 1.4T tokens -> matched GPT-3 performance
Llama-3 (70B params) trained on 15T tokens (200x ratio, overtrained for efficiency)
```

### The Scaling Law Formula

Performance (measured by loss L) follows a power law:

```
L(N, D) = A/N^alpha + B/D^beta + E

Where:
  N = number of parameters
  D = number of training tokens
  A, B = constants
  alpha ~ 0.076, beta ~ 0.095
  E = irreducible loss (entropy of natural language)
```

**What this means practically:**
- Performance improves **predictably** with scale
- You can predict the performance of a 100B model from experiments with 1B models
- There are no sudden transitions — smooth power law improvement
- The improvements continue over many orders of magnitude with no sign of slowing

### Emergent Abilities vs Smooth Scaling

There was debate about whether large models exhibit "emergent" abilities that suddenly appear at certain scales. Recent analysis suggests this is largely an artifact of measurement:
- When measured with **binary accuracy** (right or wrong), abilities appear to "emerge" suddenly
- When measured with **continuous metrics** (log-probability of correct answer), improvement is smooth
- The underlying capability improves smoothly; the binary metric just has a threshold

### Implications for Applied ML Engineers

1. **For a fixed compute budget, there is an optimal model size and data ratio.** Training a model that is too large on too little data (or too small on too much data) wastes compute.

2. **Inference cost scales with parameters.** Production systems often "overtrain" smaller models (more tokens than Chinchilla-optimal) because serving a 7B model is much cheaper than serving a 70B model, even if the 70B model is more capable.

3. **Scaling laws enable planning.** You can estimate the compute needed to reach a target performance level before spending millions on GPU clusters.

---

### Check Your Understanding

1. A decoder-only transformer uses causal masking during training. Why is -inf used before softmax rather than zeroing attention weights after softmax? What mathematical property is preserved?
2. In an encoder-decoder model like T5, cross-attention uses Q from the decoder and K, V from the encoder. Why this arrangement and not the reverse (Q from encoder, K/V from decoder)?
3. The Chinchilla scaling laws suggest optimal tokens = 20 * parameters. Yet Llama-3 was trained with a 200x ratio (far more tokens per parameter). Why would you deliberately "overtrain" a model?

<details>
<summary>Answers</summary>

1. Setting scores to -inf before softmax ensures that after exponentiation (e^(-inf) = 0), the masked positions contribute zero weight AND the remaining weights sum to exactly 1.0 through softmax's normalization. If you zeroed weights after softmax, the remaining weights would still sum to their original (non-1.0) total, requiring a separate renormalization step. Using -inf before softmax preserves the proper probability distribution automatically.
2. Cross-attention lets the decoder "ask questions" of the encoder. Q from the decoder represents "what information do I need right now to generate the next token?" K from the encoder represents "what information does each input position offer?" V from the encoder is "the actual information to retrieve." The reverse would mean the encoder queries the decoder, which makes no sense during training (the decoder output is being generated) and would leak future information.
3. Chinchilla-optimal means the best loss for a fixed compute budget. But production deployment cares about inference cost, not training cost. A 7B model trained on 15T tokens costs more to train than Chinchilla-optimal, but is much cheaper to serve than a 70B model trained on 1.4T tokens (which achieves similar quality). Since inference runs millions of times but training runs once, overtraining smaller models for deployment efficiency is economically rational.

</details>

---

## 13. Context Window: What Limits It and How It Is Being Extended

### What Limits Context Length

**Quadratic attention cost:** Standard self-attention computes a (seq_len x seq_len) attention matrix. Doubling the context length **quadruples** the computation and memory for attention.

```
4K context:   4K * 4K   = 16M entries per layer
32K context:  32K * 32K = 1B entries per layer (64x more)
128K context: 128K*128K = 16B entries per layer (1000x more)
```

**KV cache memory:** During inference, storing K and V for all previous tokens grows linearly with context length. For a 70B model at 128K context, the KV cache alone can require 10+ GB.

**Positional encoding extrapolation:** Models trained on short sequences may not generalize their position encoding to longer sequences. The attention patterns learned at position 1000 may not work at position 50000.

### How Context Is Being Extended

**Flash Attention:** Computes exact attention without materializing the full attention matrix in memory. Reduces memory from O(T^2) to O(T). This is the single most important technique for long context.

**Grouped-Query Attention (GQA):** Reduces KV cache by sharing K and V across groups of query heads. Llama-3 uses 8 KV heads for 32 query heads, reducing KV cache by 4x.

**RoPE scaling:** Extends the positional encoding to longer sequences by adjusting the rotation frequencies. NTK-aware scaling and YaRN allow models trained on 4K context to work at 128K+.

**Ring Attention:** Distributes long sequences across multiple GPUs, with each GPU computing attention for its chunk and passing KV states to the next GPU in a ring. Enables million-token contexts.

**Sparse attention:** Instead of attending to all positions, attend to a subset (local window + global tokens + random tokens). Linear complexity but loses some long-range information.

### The Landscape in 2026

| Model | Context Length | How They Do It |
|-------|---------------|----------------|
| GPT-4 Turbo | 128K | Flash Attention + rope scaling |
| Claude 3.5 | 200K | Proprietary techniques |
| Gemini 1.5 Pro | 1M+ | Ring attention + sparse attention |
| Llama-3 | 128K | GQA + RoPE scaling + Flash Attention 2 |

---

## 14. KV Cache: Speeding Up Inference

### The Problem

During autoregressive generation, we generate one token at a time. At step t, we need attention over all t previous tokens. Without caching, we recompute Q, K, V for all previous tokens at every step — O(T^2) total computation for a T-token sequence.

### The Solution

The key insight: **K and V for previous tokens do not change.** Only the new token's Q, K, V need to be computed. Cache the K and V:

```python
# At each generation step:
# 1. Compute Q, K, V for ONLY the new token
q_new = x_new @ W_Q     # (1, d_k)
k_new = x_new @ W_K     # (1, d_k)
v_new = x_new @ W_V     # (1, d_v)

# 2. Append to cache
k_cache = torch.cat([k_cache, k_new], dim=0)   # (t, d_k)
v_cache = torch.cat([v_cache, v_new], dim=0)    # (t, d_v)

# 3. Attend new query to all cached keys
scores = q_new @ k_cache.T / sqrt(d_k)   # (1, t)
weights = softmax(scores)
output = weights @ v_cache                 # (1, d_v)
```

This reduces per-step computation from O(T * d^2) to O(d^2) — a massive speedup. The total for T tokens drops from O(T^2 * d^2) to O(T * d^2).

### Memory Cost

KV cache for each layer stores `2 * seq_len * d_k_per_head * n_kv_heads` values. For a 70B model:

```
Config: 80 layers, 8 KV heads (GQA), d_k=128 per head, FP16
Per token per layer: 2 * 8 * 128 * 2 bytes = 4,096 bytes
Per token total: 4,096 * 80 layers = 327,680 bytes = 320 KB
For 4K context: 320 KB * 4,096 = 1.28 GB
For 128K context: 320 KB * 131,072 = 40 GB
```

KV cache is why serving large models is **memory-bound, not compute-bound**. The cache memory is the primary constraint on batch size and context length in production.

### Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)

| Variant | KV Heads | Query Heads | KV Cache Size | Quality |
|---------|----------|-------------|---------------|---------|
| Multi-Head (MHA) | 32 | 32 | Full | Best |
| Grouped-Query (GQA) | 8 | 32 | 1/4 of MHA | Near-MHA |
| Multi-Query (MQA) | 1 | 32 | 1/32 of MHA | Slightly worse |

GQA has become the standard because it provides nearly the same quality as full MHA while reducing KV cache by 4-8x. Llama-3 uses GQA.

---

## 15. Flash Attention: Making Long Context Practical

### The Memory Wall Problem

Standard attention materializes the full (seq_len x seq_len) attention matrix in GPU HBM (high bandwidth memory):
- 4K tokens: 4K * 4K * 2 bytes = 32 MB (fine)
- 32K tokens: 32K * 32K * 2 bytes = 2 GB (large)
- 128K tokens: 128K * 128K * 2 bytes = 32 GB (exceeds GPU memory)

Even for shorter sequences, the bottleneck is not computation but **memory bandwidth**. Moving data between HBM and SRAM (on-chip cache) takes longer than the actual arithmetic.

### How Flash Attention Works

Flash Attention (Dao et al., 2022) computes exact attention in **tiles** that fit in SRAM, never materializing the full attention matrix in HBM:

```
Algorithm:
1. Tile Q, K, V into blocks that fit in SRAM (e.g., 128 tokens per block)
2. For each Q block:
   a. For each K, V block:
      - Load K block, V block into SRAM
      - Compute local attention scores: Q_block @ K_block^T / sqrt(d_k)
      - Track running maximum and sum for online softmax
      - Accumulate partial weighted sum of V
   b. Finalize the softmax normalization
3. Write the final output to HBM
```

The "online softmax" trick is the key mathematical insight: you can compute softmax incrementally by maintaining a running maximum and sum, without ever needing all scores simultaneously.

### Results

- **Memory:** O(seq_len) instead of O(seq_len^2) — the full attention matrix is never stored
- **Speed:** 2-4x faster than standard attention due to fewer HBM reads/writes
- **Exactness:** Mathematically identical to standard attention (no approximation)

### Flash Attention 2 and 3

- **FA2:** Better parallelism across sequence length (tiles along the sequence) and heads. Achieves near-theoretical peak FLOPS utilization. 2x faster than FA1.
- **FA3:** Exploits H100 tensor cores and asynchronous pipelines. FP8 support for even lower memory and higher throughput.

### Why It Matters

Flash Attention is what makes long-context models **practical**. It is now the default in:
- PyTorch: `torch.nn.functional.scaled_dot_product_attention` auto-selects Flash Attention
- HuggingFace Transformers: `model.config.attn_implementation = "flash_attention_2"`
- vLLM, TensorRT-LLM, and all major inference engines

---

### Check Your Understanding

1. A 70B model with 80 layers, 8 KV heads (GQA), and d_k=128 per head uses 320 KB of KV cache per token. If you are serving this model with 40 GB of available memory for KV cache, what is the maximum context length you can support per request? What if you used full MHA (64 KV heads instead of 8)?
2. Flash Attention claims O(T) memory instead of O(T^2). What is the key algorithmic trick that makes this possible? Is there any approximation involved?
3. Why does GQA (sharing K/V across groups of query heads) reduce KV cache memory without significantly hurting quality?

<details>
<summary>Answers</summary>

1. With GQA (8 KV heads): 40 GB / 320 KB per token = 40 * 1024 * 1024 KB / 320 KB = ~131,072 tokens (~128K context). With full MHA (64 KV heads, 8x more): 320 KB * 8 = 2,560 KB per token. 40 GB / 2,560 KB = ~16,384 tokens (~16K context). GQA enables 8x longer context for the same memory budget.
2. The key trick is the "online softmax" algorithm: Flash Attention processes Q, K, V in tiles that fit in GPU SRAM, computing softmax incrementally by maintaining a running maximum and running sum. It never needs the full attention matrix in memory at once -- each tile's partial results are accumulated into the final output. There is zero approximation: the result is mathematically identical to standard attention. The O(T) memory comes from never materializing the T x T attention matrix.
3. GQA works because different query heads attending to the same KV representations can still learn distinct attention patterns -- they just share the same "database" (K, V) while asking different "questions" (Q). The query projections remain independent per head, so each head can focus on different aspects of the shared keys and values. Empirically, the quality loss from sharing KV across 4-8 query heads is minimal (<0.5% on benchmarks), while the 4-8x KV cache reduction is critical for serving long-context models.

</details>

---

## 16. The Complete Transformer Stack — Summary

```
Input token IDs
    |
    v
Token Embedding (nn.Embedding: vocab_size -> d_model)
    |
    v
+------------------------------------------+
| Transformer Block (repeated N times)      |
|                                           |
|  RMSNorm                                  |
|    |                                      |
|  Multi-Head Self-Attention (with RoPE)    |
|    |  + causal mask (for decoder)         |
|    |  + KV cache (during inference)       |
|    |  + Flash Attention (for efficiency)  |
|    |  + GQA (for KV cache reduction)      |
|    v                                      |
|  + residual connection                    |
|    |                                      |
|  RMSNorm                                  |
|    |                                      |
|  SwiGLU FFN (expand, gate, compress)      |
|    v                                      |
|  + residual connection                    |
+------------------------------------------+
    |
    v
Final RMSNorm
    |
    v
Linear projection (lm_head: d_model -> vocab_size)
    |
    v
Logits -> softmax -> next token probabilities
```

### Scale Reference (2026)

| Model | Layers | d_model | Heads (Q/KV) | d_ff | Params | Context |
|-------|--------|---------|-------------|------|--------|---------|
| GPT-2 Small | 12 | 768 | 12/12 | 3072 | 117M | 1024 |
| Llama-3-8B | 32 | 4096 | 32/8 | 14336 | 8B | 128K |
| Llama-3-70B | 80 | 8192 | 64/8 | 28672 | 70B | 128K |
| GPT-4 (est.) | ~120 | ~12288 | ~96/? | ~49152 | ~1.8T MoE | 128K |

The architecture is the same across all sizes. The same code handles all of them — only the config numbers change.

---

## 17. Advanced Topics: Mixture of Experts (MoE)

### The Idea

Instead of one FFN, have multiple FFN "experts" and a router that selects which experts to use for each token:

```python
# Standard transformer: every token goes through the same FFN
output = ffn(x)

# MoE: router selects top-k experts for each token
router_logits = x @ W_router       # (seq_len, n_experts)
top_k_indices = topk(router_logits, k=2)  # Select 2 experts per token
output = sum(expert_i(x) * gate_i for i in top_k_indices)
```

### Why MoE Matters

A 1.8T parameter MoE model activates only ~200B parameters per token (the selected experts). This means:
- **Training cost:** Proportional to total parameters (all experts updated)
- **Inference cost:** Proportional to active parameters (only selected experts run)
- **Knowledge capacity:** Full parameter count (all experts store different knowledge)

MoE gives you the knowledge capacity of a huge model with the inference cost of a smaller model. GPT-4 and Mixtral use MoE.

### Challenges

- **Load balancing:** If all tokens route to the same experts, other experts waste memory. Auxiliary losses encourage balanced routing.
- **Communication overhead:** In distributed training, experts on different GPUs require all-to-all communication.
- **Memory:** All experts must be in memory, even though only k are active per token.

---

## Common Pitfalls

**Pitfall 1: Forgetting the causal mask during decoder training**
- Symptom: Model achieves suspiciously high accuracy during training but generates gibberish during inference
- Why: Without the causal mask, each position can see future tokens during training, making next-token prediction trivially easy. But during inference, future tokens do not exist, so the model has never learned to predict without them
- Fix: Always apply the lower-triangular causal mask before softmax in decoder self-attention. Verify by checking that training loss starts at -ln(1/vocab_size) (random), not near zero

**Pitfall 2: Not using Flash Attention for sequences longer than a few thousand tokens**
- Symptom: Out of memory errors or extremely slow training on long sequences, even with a powerful GPU
- Why: Standard attention materializes an O(T^2) attention matrix. For 32K tokens, this is 32K * 32K * 2 bytes = 2 GB per head per layer -- far more than the actual model weights
- Fix: Use Flash Attention (via `torch.nn.functional.scaled_dot_product_attention` in PyTorch, or `attn_implementation="flash_attention_2"` in HuggingFace). It is mathematically identical but uses O(T) memory

**Pitfall 3: Confusing encoder-only, decoder-only, and encoder-decoder architectures**
- Symptom: Using BERT (encoder-only) for text generation, or GPT (decoder-only) for sentence embeddings without understanding the limitations
- Why: Encoder-only models use bidirectional attention and produce embeddings (not generated text). Decoder-only models use causal attention and generate text autoregressively. Using the wrong architecture for the task leads to poor results
- Fix: For classification/retrieval/embeddings, use encoder-only (BERT, sentence-transformers). For generation, use decoder-only (GPT). For input-to-output transformation (translation), use encoder-decoder (T5) or decoder-only with appropriate prompting

**Pitfall 4: Ignoring KV cache memory when planning inference deployment**
- Symptom: Model fits in GPU memory but crashes or throttles when serving long-context requests or multiple concurrent users
- Why: KV cache grows linearly with context length and number of concurrent requests. A 70B model at 128K context can need 40+ GB just for KV cache, on top of the model weights
- Fix: Use GQA to reduce KV cache by 4-8x, INT8/FP8 quantization for cache values, and paged attention (vLLM) for efficient memory management across requests. Always calculate KV cache requirements before deployment

## Hands-On Exercises

### Exercise 1: Implement Self-Attention from Scratch
**Goal:** Deeply understand the attention computation by implementing it without using any attention library functions
**Task:**
1. Implement single-head scaled dot-product attention in PyTorch: take input X (batch, seq_len, d_model), compute Q, K, V projections, compute QK^T/sqrt(d_k), apply causal mask, softmax, and multiply by V
2. Extend to multi-head attention: split into h heads, run attention on each, concatenate, and project through W_O
3. Test on a small sequence and verify your attention weights match `torch.nn.functional.scaled_dot_product_attention`
4. Visualize the attention matrix for a sample sentence -- which tokens attend to which?
**Verify:** Your implementation's output should match PyTorch's built-in attention to within floating-point precision (atol=1e-5). The causal mask should produce a lower-triangular attention pattern.

### Exercise 2: Compare Positional Encoding Schemes
**Goal:** Understand how different positional encodings affect a transformer's ability to handle position
**Task:**
1. Build a small transformer (2 layers, 4 heads, d_model=128) for a sequence copying task (input a sequence, output the same sequence in reverse)
2. Train three versions: (a) sinusoidal positional encoding, (b) learned positional embeddings, (c) no positional encoding
3. Test each on sequences of the training length (e.g., 50 tokens) and longer sequences (100, 200 tokens)
4. Plot accuracy vs sequence length for all three versions
**Verify:** Without positional encoding, the model should fail entirely (random output). Sinusoidal should generalize better to longer sequences than learned embeddings (which have a hard length limit). Both should succeed at the training length.

---

## 18. Interview Questions

### Conceptual

1. **Walk through the complete attention computation with shapes.** Start with input X (seq_len, d_model), project to Q, K, V using weight matrices, compute QK^T (seq_len x seq_len), divide by sqrt(d_k), apply causal mask (set future positions to -inf), softmax each row, multiply by V to get output (seq_len, d_v). The softmax-normalized attention matrix tells you how much each position attends to every other position.

2. **Why divide by sqrt(d_k)?** The dot product's variance grows linearly with d_k. Without scaling, large dot products push softmax into saturation (near-one-hot), killing gradient flow. Dividing by sqrt(d_k) normalizes variance to 1, keeping softmax in its informative gradient region.

3. **Explain multi-head attention and why it is better than single-head.** Multiple heads let the model learn different relationship types (syntactic, semantic, positional) in parallel. Each head operates in a lower-dimensional subspace (d_k = d_model/h) so total cost is similar to single-head at full dimension. The W_O projection combines insights from all heads.

4. **What is RoPE and why is it preferred over learned positional embeddings?** RoPE rotates Q and K vectors by position-dependent angles. The dot product of rotated Q and K depends on relative position (the absolute positions cancel out). This gives the model relative position information without explicit position matrices, uses zero parameters, and can extrapolate to longer sequences via frequency scaling.

5. **Explain the KV cache and why GQA reduces its memory cost.** During autoregressive generation, K and V from previous tokens are cached to avoid recomputation. KV cache grows linearly with context length and is the main memory bottleneck. GQA shares K and V across groups of query heads (e.g., 8 KV heads for 32 query heads), reducing cache by the group factor.

6. **What is Flash Attention and how does it achieve linear memory?** Flash Attention tiles the attention computation into blocks that fit in GPU SRAM, computing exact attention without ever materializing the full quadratic attention matrix in HBM. It uses the online softmax trick to accumulate results incrementally. This reduces memory from O(T^2) to O(T) and achieves 2-4x speedup by being compute-bound instead of memory-bound.

7. **Pre-norm vs post-norm: which is better and why?** Pre-norm (norm before sub-layer) is universally preferred for modern LLMs. The residual path is completely clean (no normalization in the identity path), which provides unbounded gradient flow. This makes training more stable for deep models and eliminates the need for careful warmup. Post-norm has slightly better theoretical final performance but is much harder to train.

### System Design

8. **You need to serve a 70B parameter model for production chat. What are the main bottlenecks and how do you address them?** Memory (KV cache + model weights): use GQA, quantization (INT4/FP8), and tensor parallelism across GPUs. Latency: prefill phase is compute-bound (use Flash Attention), decode phase is memory-bandwidth-bound (use speculative decoding, continuous batching). Cost: serve multiple requests simultaneously with paged attention (vLLM).

9. **Design a retrieval-augmented generation (RAG) system using transformers.** Encode documents with a bi-encoder (sentence-transformer/E5 — encoder-only). Store embeddings in a vector database. At query time, encode the query with the same encoder, retrieve top-k relevant documents, prepend them to the prompt, and generate with a decoder-only LLM. Key decisions: chunk size (512-1024 tokens), reranker (cross-encoder for precision), and context window management (most relevant chunks first).

10. **Explain the difference between encoder-only, decoder-only, and encoder-decoder architectures.** Encoder-only (BERT): bidirectional attention, for understanding tasks (classification, retrieval). Decoder-only (GPT): causal attention, for generation. Encoder-decoder (T5): encoder understands input bidirectionally, decoder generates output autoregressively with cross-attention to the encoder. Decoder-only dominates because autoregressive generation is universal and the architecture scales most efficiently.

---

## Key Takeaways

1. Self-attention lets every token directly interact with every other token in O(1) path length — the fundamental advance over RNNs
2. Q, K, V projections let each token play three roles: seeker, advertiser, and provider of information
3. Scaling by sqrt(d_k) prevents softmax saturation and maintains healthy gradient flow
4. Multi-head attention learns multiple relationship types in parallel (syntactic, semantic, positional, induction)
5. RoPE encodes relative position via rotation — the modern standard for positional encoding
6. The FFN stores factual knowledge; attention routes information between positions
7. Pre-norm with RMSNorm + residual connections enables stable training of very deep models
8. Decoder-only with causal masking has become the dominant paradigm because next-token prediction is a universal training signal
9. KV cache is the main inference bottleneck; GQA and quantization are the primary mitigations
10. Flash Attention makes long context practical by tiling attention into SRAM without materializing the full attention matrix
11. Scaling laws show predictable improvement with more parameters and data — no plateaus in sight
12. The architecture is simple. Scale and data are what separate a toy model from a frontier model.

## Summary

The transformer's power comes from self-attention, which lets every position directly interact with every other position in O(1) path length -- eliminating the information bottleneck and sequential processing that limited RNNs. The architecture is remarkably simple: learned Q, K, V projections, scaled dot-product attention, multi-head parallelism, residual connections, and position-wise FFNs. Everything else (RoPE, RMSNorm, SwiGLU, GQA, Flash Attention, KV cache) is engineering optimization on this foundation. The decoder-only variant with causal masking and next-token prediction has emerged as the universal architecture because generation subsumes all other tasks.

## What's Next

- **Builds on this:** Large language model fine-tuning, RLHF, and prompt engineering all build directly on the transformer architecture covered here
- **Related:** [CNNs](../cnns/COURSE.md) -- Vision Transformers apply self-attention to image patches, and understanding both CNN inductive biases and transformer flexibility helps you choose the right vision architecture for your constraints
