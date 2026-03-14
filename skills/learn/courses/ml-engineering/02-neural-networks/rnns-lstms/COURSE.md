## Learning Objectives

By the end of this lesson, you will:

**Understand:**
- How RNNs process sequential data by maintaining a hidden state, and why the vanishing gradient problem (caused by repeated multiplication of the same weight matrix) limits vanilla RNNs to ~20 timesteps of effective memory
- How LSTMs solve vanishing gradients with a cell state that uses additive updates and learned gates (forget, input, output), and how GRUs simplify this to two gates with comparable performance
- Why transformers replaced RNNs for most tasks (parallelization, O(1) long-range connections, scaling efficiency), and where RNNs remain relevant (streaming, edge devices, state space models)

**Apply:**
- Implement LSTM-based models in PyTorch for classification and sequence labeling, including proper handling of variable-length sequences with pack_padded_sequence
- Configure RNN training with gradient clipping, proper forget gate bias initialization, and bidirectional processing

**Analyze:**
- Evaluate whether an RNN, transformer, or state space model is the right architecture for a given sequential task, considering data size, latency, memory constraints, and deployment environment

## Prerequisites

- **Layers and activations** -- RNNs use the same fundamental building blocks (linear transformations, activations, learnable weights) as feedforward networks (see [Fundamentals](../fundamentals/COURSE.md))
- **Backpropagation** -- understanding how gradients flow backward through layers is essential for grasping why gradients vanish in RNNs (see [Training Mechanics](../training-mechanics/COURSE.md))
- **Chain rule for backpropagation through time** -- BPTT is the chain rule applied across timesteps, requiring comfort with repeated application of derivatives (see [Calculus](../../01-math-foundations/calculus/COURSE.md))

---

# Recurrent Neural Networks, LSTMs, and GRUs

## Why This Lesson Matters

RNNs and LSTMs are the architectures that transformers replaced. Understanding them deeply is important for three reasons: (1) you will encounter them in legacy systems and specialized applications, (2) the problems they tried to solve — sequential processing, memory, long-range dependencies — are the same problems transformers solve better, and understanding the failure modes of RNNs illuminates why transformers work, (3) modern state space models (Mamba, RWKV) are spiritual successors of RNNs, and the concepts transfer directly.

---

## 1. RNNs: Processing Sequences

### The Problem with Feedforward Networks

A standard feedforward network processes a fixed-size input and produces a fixed-size output. But sequences (text, audio, time series, video) have two properties that feedforward networks cannot handle:

1. **Variable length:** Sentences can be 3 words or 300 words
2. **Temporal dependencies:** The meaning of a word depends on what came before it. "I went to the bank to deposit money" vs "I went to the bank to fish"

### The Core Idea

An RNN processes one element at a time, maintaining a **hidden state** that carries information from previous timesteps:

```python
# At each timestep t:
h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t + b_h)
y_t = W_hy @ h_t + b_y
```

The hidden state `h_t` is a compressed summary of everything the network has seen so far. It is updated at each step by combining the previous hidden state with the current input.

```
UNROLLED RNN:

h_0 -> [RNN Cell] -> h_1 -> [RNN Cell] -> h_2 -> [RNN Cell] -> h_3 -> output
            ^                    ^                    ^
            |                    |                    |
           x_1                  x_2                  x_3

The SAME weights (W_hh, W_xh) are used at every timestep.
```

### Weight Sharing Across Time

Just as CNNs share weights across spatial positions, RNNs share weights across time steps. The same weight matrices `W_hh`, `W_xh` are used at every timestep. This means:
- The RNN can process sequences of any length with a fixed number of parameters
- It learns **temporal patterns** that apply regardless of when they occur in the sequence
- The number of parameters does not grow with sequence length

### What the Hidden State Represents

The hidden state is a **vector** (typically 128-1024 dimensions) that encodes everything the model "remembers" about the sequence so far. It is a lossy compression — the network must learn what information to keep and what to discard.

For language modeling, the hidden state after processing "The cat sat on the" should encode enough information to predict "mat" — knowledge about syntax (noun expected after article), semantics (location context), and specific content (cats sitting on things).

### Types of RNN Tasks

```
One-to-one:     Standard feedforward (not really an RNN)
One-to-many:    Image captioning (image -> sequence of words)
Many-to-one:    Sentiment analysis (sequence of words -> label)
Many-to-many:   Machine translation (sequence -> sequence, different lengths)
Many-to-many:   Named entity recognition (each input position gets a label)
```

---

## 2. The Vanishing Gradient Problem

### Why Vanilla RNNs Fail on Long Sequences

This is the fundamental limitation that motivated LSTMs, GRUs, and ultimately transformers.

When you unroll an RNN across time, it becomes a very deep network where gradients must flow backward through many timesteps. At each timestep, the gradient is multiplied by the weight matrix `W_hh` and the derivative of tanh:

```
d_loss/d_h_1 = d_loss/d_h_T * product(t=2 to T) of [W_hh * tanh'(h_t)]
```

The tanh derivative is always between 0 and 1 (maximum value is 1 at tanh(0)). The product of many numbers less than 1 goes to zero exponentially:

```
After 10 timesteps:  0.7^10 = 0.028
After 20 timesteps:  0.7^20 = 0.0008
After 50 timesteps:  0.7^50 = 0.0000018
```

**After roughly 10-20 timesteps, gradients effectively vanish.** The network cannot learn dependencies longer than about 20 tokens.

### Concrete Example

Consider the sentence: "The cat, which was sitting on the mat near the window overlooking the garden where the flowers were blooming beautifully in the warm spring sunshine, **purred**."

An RNN needs to connect "cat" (subject) to "purred" (verb) across ~25 tokens. With vanilla RNNs, the gradient signal from "purred" back to "cat" is essentially zero. The network cannot learn this subject-verb agreement.

### Why This Is Worse Than in Feedforward Networks

In feedforward networks, each layer has **different** weights, so some layers might amplify gradients while others shrink them, partially canceling out. In RNNs, the **same** weight matrix is multiplied over and over:

- If its largest singular value is < 1: gradients vanish systematically
- If its largest singular value is > 1: gradients explode systematically
- There is a very narrow band where things work, and it is practically impossible to stay in

### The Exploding Gradient Side

When gradients explode (singular value > 1), the updates become enormous and training diverges (loss goes to NaN). The fix is **gradient clipping**: cap the norm of the gradient vector before applying it.

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

This is standard practice for any RNN training and is essentially required. Gradient clipping solves exploding gradients trivially — vanishing gradients are the hard problem.

---

## 3. LSTMs: Long Short-Term Memory

### The Key Insight

LSTMs (Hochreiter and Schmidhuber, 1997) introduce a **cell state** — a separate memory channel that runs through the entire sequence with only **additive** interactions. Information can be added to or removed from the cell state through learned **gates**.

**Analogy:** Think of a conveyor belt running through a factory. Items ride the belt unchanged unless a worker at a station explicitly adds something or removes something. The belt itself does not transform anything — it just carries. This is the cell state.

The RNN hidden state, by contrast, is like passing a message through a chain of people — each person transforms it, and by the end, the original message is distorted beyond recognition.

### Architecture Diagram

```
LSTM CELL:

                 Cell State (c_{t-1}) ----[x forget]----[+ new info]----> c_t
                                              ^               ^
                                              |               |
              +------- Forget Gate ------+    |               |
              |                          |    |               |
x_t -------->|  Concatenate [h_{t-1}, x_t]   |               |
h_{t-1} ---->|                          |    |               |
              |------- Input Gate -------+----|               |
              |                          |                    |
              |------- Candidate --------+--------------------+
              |                          |
              |------- Output Gate ------+----> h_t (output)
              +--------------------------+
```

### The Three Gates

Each gate is a sigmoid layer (outputs 0-1) that controls information flow:

#### Forget Gate: "What should I erase from memory?"

```python
f_t = sigmoid(W_f @ [h_{t-1}, x_t] + b_f)
# Output: vector of values between 0 and 1 for each dimension of cell state
# 0 = completely forget this dimension
# 1 = completely keep this dimension
```

**Example:** When reading a story, the forget gate learns to erase the previous subject when a new subject appears. "The cat purred. The dog..." — forget gate erases "cat" information and prepares for "dog" information.

#### Input Gate: "What new information should I store?"

```python
i_t = sigmoid(W_i @ [h_{t-1}, x_t] + b_i)    # What dimensions to update (0-1)
c_tilde = tanh(W_c @ [h_{t-1}, x_t] + b_c)    # Candidate new values (-1 to 1)
```

The input gate has two parts:
1. **Which dimensions to update** (sigmoid decides 0-1 per dimension)
2. **What the new values should be** (tanh creates candidate values)

Together: `i_t * c_tilde` is the new information to write to memory.

#### Cell State Update — The Critical Step

```python
c_t = f_t * c_{t-1} + i_t * c_tilde
#     ^-- forget old    ^-- add new
```

This is the conveyor belt in action. The cell state update is **additive** — no matrix multiplication of the cell state by a weight matrix. This is exactly why gradients do not vanish: the gradient flows through addition (gradient = 1, modified only by the forget gate) rather than through repeated matrix multiplication.

#### Output Gate: "What should I expose from memory?"

```python
o_t = sigmoid(W_o @ [h_{t-1}, x_t] + b_o)
h_t = o_t * tanh(c_t)
```

Not everything in memory needs to be relevant to the current output. The output gate filters the cell state before exposing it as the hidden state.

**Example:** You know the character's name, age, and location, but the current sentence only requires their location. The output gate exposes only the location.

### How LSTMs Solve Vanishing Gradients

The gradient through the cell state is:

```
d_c_t / d_c_{t-1} = f_t    (the forget gate value)
```

The forget gate `f_t` is a **learned** sigmoid. It can be close to 1 when information needs to persist across many timesteps. Unlike the fixed multiplication by `W_hh * tanh'` in vanilla RNNs, the LSTM **learns** how much gradient to propagate.

When the forget gate is 1 (keeping everything):
- The gradient passes through perfectly, regardless of sequence length
- The network learns long-range dependencies by learning to keep the forget gate open

When the forget gate is 0 (forgetting):
- The gradient is blocked, but intentionally — the network learned that this information is no longer relevant

### Critical Implementation Detail

**Always initialize the forget gate bias to a positive value (1.0 or 2.0).** This starts the forget gate biased toward "keep everything," so information flows freely by default. Without this, LSTMs often fail to learn long-range dependencies because the randomly initialized forget gates start near 0.5, which halves the gradient at every step.

```python
# After creating the LSTM:
for name, param in lstm.named_parameters():
    if 'bias' in name:
        # Bias is organized as [input_gate, forget_gate, cell_gate, output_gate]
        n = param.size(0)
        param.data[n//4:n//2].fill_(1.0)  # Set forget gate bias to 1.0
```

### Stacking LSTMs

You can stack multiple LSTM layers. The output sequence of one becomes the input to the next:

```python
lstm = nn.LSTM(
    input_size=128,      # Dimension of input vectors
    hidden_size=256,     # Dimension of hidden state
    num_layers=2,        # Stack 2 LSTM layers
    batch_first=True,    # Input shape: (batch, seq_len, features)
    dropout=0.2,         # Dropout between layers (not within)
    bidirectional=True   # Process forward and backward
)

# Input:  (batch, seq_len, input_size) = (32, 100, 128)
# Output: (batch, seq_len, 2*hidden_size) = (32, 100, 512)  # bidirectional doubles
# h_n:    (2*num_layers, batch, hidden_size) = (4, 32, 256)  # final hidden states
# c_n:    (2*num_layers, batch, hidden_size) = (4, 32, 256)  # final cell states
```

---

### Check Your Understanding

1. Why does the LSTM cell state use additive updates (c_t = f_t * c_{t-1} + i_t * c_tilde) rather than multiplicative updates? How does this specifically address the vanishing gradient problem?
2. You are training an LSTM and notice it fails to learn long-range dependencies (e.g., connecting a subject to a verb 30 tokens later). You have not modified the default PyTorch initialization. What is the most likely cause, and what is the one-line fix?
3. An LSTM has separate cell state and hidden state. Why not just use the cell state directly as the output? What role does the output gate play?

<details>
<summary>Answers</summary>

1. With additive updates, the gradient of c_t with respect to c_{t-1} is simply f_t (the forget gate value), not a product of weight matrices. The forget gate can learn to be close to 1 when information needs to persist, allowing gradients to flow through unchanged across many timesteps. With multiplicative updates (as in vanilla RNNs), the same weight matrix is multiplied at every step, causing systematic exponential decay or explosion.
2. The most likely cause is that the forget gate bias is initialized near zero (PyTorch default), causing the forget gate sigmoid to output ~0.5, which halves the gradient at every timestep. Fix: set the forget gate bias to 1.0 with `lstm.bias_ih_l0.data[hidden_size:2*hidden_size].fill_(1.0)` (and similarly for bias_hh). This biases the gate toward "keep everything" by default.
3. The cell state stores everything the LSTM has chosen to remember, but not all of it is relevant to the current output. The output gate filters the cell state, exposing only the portions relevant to the current context. For example, the cell state might store a character's name, age, and location, but if the current sentence only needs the location, the output gate exposes only that information. Without the output gate, the hidden state would be an unfiltered dump of all stored information.

</details>

---

## 4. GRUs: Gated Recurrent Units

### Simplified Gating

GRUs (Cho et al., 2014) simplify the LSTM by:
- **Merging** the forget and input gates into a single **update gate**
- **Merging** the cell state and hidden state into one vector
- Removing the output gate

```python
# Update gate: how much of the new state to mix in (vs keep old)
z_t = sigmoid(W_z @ [h_{t-1}, x_t])

# Reset gate: how much of the previous state to expose to the candidate
r_t = sigmoid(W_r @ [h_{t-1}, x_t])

# Candidate hidden state (using reset-gated previous state)
h_tilde = tanh(W @ [r_t * h_{t-1}, x_t])

# Final hidden state: interpolate between old and new
h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde
```

### Key Difference from LSTM

The update gate `z_t` directly interpolates between the old and new states. When `z_t = 0`, the hidden state is unchanged (perfect memory). When `z_t = 1`, the hidden state is completely replaced.

The clever trick: the **same** gate controls both forgetting and updating. If `z_t` says "update a lot" (high value), then `(1 - z_t)` says "forget a lot." This coupling means the GRU has fewer degrees of freedom than LSTM but also fewer parameters to learn.

### GRU vs LSTM Comparison

| Aspect | LSTM | GRU |
|--------|------|-----|
| Parameters per cell | 4 * (d_h^2 + d_h * d_x + d_h) | 3 * (d_h^2 + d_h * d_x + d_h) |
| Gates | 3 (forget, input, output) | 2 (update, reset) |
| Memory mechanism | Separate cell state + hidden state | Single hidden state |
| Long-range performance | Slightly better empirically | Comparable on most tasks |
| Training speed | Slower (~25% more computation) | Faster |
| When to use | Very long sequences, proven track record | Shorter sequences, faster iteration |

### Practical Guidance

In practice, the choice between LSTM and GRU rarely makes a significant difference (typically <1% accuracy). If you are using a recurrent architecture:
- Start with GRU (simpler, faster to train)
- Switch to LSTM only if you see evidence of insufficient memory over long sequences
- But first ask: should you be using a transformer instead?

---

## 5. Bidirectional RNNs

### The Concept

Process the sequence in both directions and concatenate the hidden states:

```python
# Forward pass:  reads x_1, x_2, ..., x_T
h_f_t = RNN_forward(x_t, h_f_{t-1})

# Backward pass: reads x_T, x_{T-1}, ..., x_1
h_b_t = RNN_backward(x_t, h_b_{t+1})

# Combined representation at each position:
h_t = [h_f_t ; h_b_t]    # concatenation, doubles the dimension
```

### Why It Helps

For many tasks, the meaning of a word depends on what comes **after** it:
- "He **read** the book" (past tense, rhymes with "red")
- "He likes to **read**" (infinitive, rhymes with "reed")
- Named entity recognition: "Apple launched..." (company) vs "apple pie..." (fruit)

A forward-only RNN at position t has only seen x_1 through x_t. A bidirectional RNN at position t has seen the **entire sequence**, giving it full context.

### Limitation

Bidirectional RNNs **cannot be used for autoregressive generation** (predicting the next token). During generation, future tokens do not exist yet, so the backward pass is impossible. This limits BiRNNs to **understanding** tasks (classification, NER, embedding).

This is the same distinction as BERT (bidirectional, for understanding) vs GPT (unidirectional, for generation).

---

## 6. Sequence-to-Sequence: Encoder-Decoder

### The Architecture

For tasks where input and output sequences have different lengths (translation, summarization, question answering), the encoder-decoder pattern was the standard before transformers:

```
ENCODER (reads input, compresses to context vector):
"The cat sat" -> [RNN] -> [RNN] -> [RNN] -> [context_vector]

DECODER (generates output from context vector):
                                    [context_vector]
                                         |
                                    [RNN] -> "Le"
                                    [RNN] -> "chat"
                                    [RNN] -> "assis"
```

The encoder reads the input sequence and compresses it into a fixed-size **context vector** (the final hidden state). The decoder generates the output sequence, conditioned on this context vector.

### The Information Bottleneck Problem

The context vector must compress the entire input into a single vector. For long sequences, this is a brutal bottleneck. A 100-word paragraph squeezed into 512 numbers inevitably loses information.

Symptoms of the bottleneck:
- Performance degrades sharply on longer inputs
- The model "forgets" information from the beginning of long inputs
- Translation quality drops for sentences longer than ~20 words

### Attention Mechanism (Bahdanau, 2014) — The Solution

Instead of one context vector, the decoder looks at **all** encoder hidden states at each step, focusing on the most relevant ones:

```python
# At decoder timestep t:
# Score each encoder state by how relevant it is
scores = decoder_h_t @ encoder_states.T        # (1, T_enc)
attention_weights = softmax(scores)              # (1, T_enc) -- probabilities
context = attention_weights @ encoder_states     # (1, d_h) -- weighted sum
```

At each decoding step, the model learns **where to look** in the input. When generating "chat" (French for "cat"), the attention focuses on the encoder state for "cat." When generating "assis" (French for "sat"), the attention shifts to "sat."

**This attention mechanism — letting the decoder "look back" at the input — was the direct precursor to the transformer's self-attention.** The key insight (every output position should dynamically attend to all input positions) is what Vaswani et al. generalized into the full self-attention mechanism.

### Teacher Forcing

During training, the decoder receives the **ground truth** previous token as input (not its own prediction). This is called teacher forcing and dramatically speeds up training by providing a stable input at each step.

**The problem:** During inference, there is no ground truth — the model uses its own predictions, which may be wrong. Errors compound. This mismatch between training and inference is called **exposure bias.**

**Solutions:**
- **Scheduled sampling:** During training, randomly use model predictions (instead of ground truth) with increasing probability as training progresses
- **Sequence-level training:** Optimize directly for sequence metrics like BLEU using reinforcement learning techniques

---

## 7. Why Transformers Replaced RNNs

### Problem 1: Sequential Processing Cannot Be Parallelized

RNNs are inherently **sequential**: you cannot compute `h_t` without first computing `h_{t-1}`. On a GPU with thousands of cores, most cores sit idle while the RNN processes one timestep at a time.

Transformers process all positions **simultaneously**. Self-attention computes relationships between all pairs of positions in one parallel matrix multiplication. On modern GPUs, this parallelism translates to 10-100x speedups for training.

```
RNN training:   T sequential steps, each O(d^2). Total: O(T * d^2), sequential
Transformer:    One parallel step of O(T^2 * d). But fully parallelized across T and d
                For typical T < 4096 and d = 4096, the transformer is far faster on GPUs
```

### Problem 2: Long-Range Dependencies Require Long Gradient Paths

In an RNN, information from token 1 must pass through **every intermediate hidden state** to reach token 50. At each step, the information is transformed (and potentially corrupted). The path length is O(T).

In a transformer, token 1 can **directly attend** to token 50 in a single operation. The path length is O(1). Long-range dependencies are trivially easy to learn because there is no information decay along the path.

### Problem 3: Fixed-Size Hidden State is a Bottleneck

An RNN's memory is its hidden state — a fixed-size vector (typically 256-1024 dimensions). It must decide what to remember and what to forget as it processes the sequence. This is inherently lossy.

A transformer's "memory" is the **full sequence** of embeddings, directly accessible via attention. Nothing is forgotten unless the context window is exceeded. Context windows of 128K+ tokens are possible because every token is explicitly stored and addressable.

### Problem 4: Scaling

Transformers scale more efficiently with compute:
- **RNNs:** More compute means longer training, but the sequential bottleneck remains. You cannot parallelize within a sequence.
- **Transformers:** More compute means bigger models, longer sequences, and faster training — all dimensions that GPUs can parallelize.

This scaling advantage is why every frontier model (GPT-4, Claude, Gemini, Llama) is a transformer.

### The Numbers Tell the Story

| Metric | LSTM (2017 SOTA) | Transformer (2017) | Transformer (2024) |
|--------|-----------------|-------------------|-------------------|
| Machine Translation (BLEU) | 26.0 | 28.4 | 35+ |
| Language Modeling (perplexity) | 58.3 | 18.7 | <5 |
| Training time for equivalent quality | 3 weeks | 3.5 days | Hours |
| Max effective context | ~200 tokens | 512 tokens | 128K+ tokens |

---

### Check Your Understanding

1. Why cannot RNN computation be parallelized across timesteps, and how does this compare to self-attention? Be specific about the data dependency.
2. The encoder-decoder architecture with attention (Bahdanau, 2014) was the direct precursor to transformers. What was the key insight from this model that transformers generalized?
3. A colleague argues that bidirectional LSTMs are always better than unidirectional LSTMs. Give a specific task where a bidirectional LSTM cannot be used and explain why.

<details>
<summary>Answers</summary>

1. In an RNN, computing h_t requires h_{t-1} as input, creating a strict sequential dependency: each timestep must wait for the previous one to complete. On a GPU with thousands of cores, most sit idle. In self-attention, each position's output depends on all input positions through matrix multiplication (Q@K^T and weights@V), but these are independent computations across positions that can be fully parallelized. There is no sequential dependency between positions.
2. The key insight was that the decoder should dynamically attend to different parts of the encoder output at each generation step, rather than compressing the entire input into a single fixed-size context vector. This "look back at all inputs" mechanism -- where attention weights are computed as compatibility scores between the decoder state and all encoder states -- is exactly what Vaswani et al. generalized into self-attention (where every position attends to every other position, not just decoder-to-encoder).
3. Autoregressive text generation (e.g., language modeling, chatbots). During generation, future tokens do not exist yet, so the backward pass of a bidirectional LSTM has no data to process. Bidirectional LSTMs require the complete sequence to be available before processing, limiting them to understanding tasks (classification, NER, embeddings) where the full input is known upfront. This is the same distinction as BERT (bidirectional) vs GPT (unidirectional).

</details>

---

## 8. When RNNs/LSTMs Are Still Used in 2026

Despite transformer dominance, RNNs have legitimate use cases:

### Edge Devices and Real-Time Streaming

RNNs process one token at a time with **constant memory**. A transformer processing a growing sequence requires a KV cache that grows linearly with sequence length.

For streaming audio on a phone, an RNN uses the same memory at minute 1 and minute 60. A transformer's KV cache grows continuously, eventually exceeding device memory.

### Very Long Sequences with Fixed Memory Budget

If you need to process a continuous stream (sensor data, live audio, IoT telemetry) without a fixed context window, RNNs naturally handle this. The hidden state is a constant-size summary, regardless of how long the stream runs.

### Tiny Models for Embedded Systems

For microcontrollers with kilobytes of RAM (smart watches, IoT sensors), a small LSTM (hidden size 32-64) can run inference with negligible memory. The smallest practical transformer still needs more resources.

### Time Series Forecasting

Simple LSTM/GRU models remain competitive for univariate time series forecasting, especially when:
- Data is limited (thousands of samples, not millions)
- The time series has clear recurrent patterns (seasonality, trends)
- Inference latency matters more than accuracy
- The system needs to process truly streaming data point-by-point

### State Space Models: The Modern Successor

Mamba, RWKV, and other state space models (SSMs) are architecturally similar to RNNs — they maintain a hidden state and can process tokens sequentially. But they are designed to be parallelizable during training:

**Training:** The recurrence can be rewritten as a convolution, enabling full parallelization across the sequence (like a transformer)

**Inference:** Process one token at a time with constant memory (like an RNN)

SSMs combine the best of both worlds. Whether they will complement or replace transformers is an active research question in 2026, but they represent the spiritual successor of RNNs — not the LSTM architectures of 2014, but the same fundamental idea of maintaining a compressed state.

### When You Should NOT Use RNNs

- **NLP tasks:** Use transformers. The accuracy gap is too large to justify RNNs.
- **Any task with a pretrained transformer available:** Fine-tuning BERT or GPT is almost always better than training an LSTM from scratch, even for small datasets.
- **When you need interpretability via attention weights:** Attention weights provide a (imperfect but useful) view into what the model focuses on. RNN hidden states are opaque.
- **When you have large-scale data:** Transformers absorb more data more effectively.

---

## 9. Practical Patterns and Tips

### Common LSTM Patterns in Code

```python
# Pattern 1: Many-to-one (classification)
lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
output, (h_n, c_n) = lstm(x)  # output: all timesteps, h_n: final hidden
logits = classifier(h_n[-1])   # Use final hidden state for classification

# Pattern 2: Many-to-many (sequence labeling)
lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
output, _ = lstm(x)            # output: (batch, seq_len, 2*hidden_size)
logits = classifier(output)    # Classify each position

# Pattern 3: Seq2seq with attention
encoder_outputs, (h_n, c_n) = encoder_lstm(src)
# At each decoder step:
decoder_output, (h_n, c_n) = decoder_lstm(tgt_token, (h_n, c_n))
attn_weights = softmax(decoder_output @ encoder_outputs.T)
context = attn_weights @ encoder_outputs
```

### Variable-Length Sequences — Padding and Packing

Real sequences have different lengths. You must handle this:

```python
# Padding: add zeros to make all sequences the same length
# Problem: the LSTM processes the padding as real data

# Solution: pack_padded_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

packed = pack_padded_sequence(padded_input, lengths, batch_first=True, enforce_sorted=False)
output, (h_n, c_n) = lstm(packed)
output, _ = pad_packed_sequence(output, batch_first=True)
```

This tells the LSTM to skip padding positions, giving you correct hidden states and saving computation.

### Training RNNs: Essential Practices

1. **Gradient clipping is mandatory.** Always clip gradient norms to 1.0-5.0.
2. **Initialize forget gate bias to 1.0+.** Default random initialization cripples long-range learning.
3. **Use bidirectional for understanding tasks.** Free accuracy boost.
4. **Dropout between layers, not within layers.** Use `nn.LSTM(dropout=0.2)`, not dropout on the hidden state.
5. **Start with GRU.** Switch to LSTM only if GRU proves insufficient.
6. **Use teacher forcing during training.** Scheduled sampling for better generalization.

---

## Common Pitfalls

**Pitfall 1: Not using gradient clipping with RNNs**
- Symptom: Loss suddenly jumps to NaN or infinity after training smoothly for a while
- Why: RNNs are inherently prone to exploding gradients because the same weight matrix is multiplied at every timestep. A single pathological sequence can produce gradient norms in the thousands
- Fix: Always use gradient clipping for RNN training. `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` is mandatory, not optional

**Pitfall 2: Not using pack_padded_sequence for variable-length sequences**
- Symptom: Model performance is poor, especially on shorter sequences; hidden states seem contaminated
- Why: Without packing, the LSTM processes padding tokens as real data, corrupting hidden states with meaningless zero-input computations. The final hidden state for a short sequence includes processing of many padding positions
- Fix: Use `pack_padded_sequence` before feeding to the LSTM and `pad_packed_sequence` after. This tells PyTorch to skip padding positions entirely

**Pitfall 3: Using an LSTM when a transformer would be better**
- Symptom: Spending weeks tuning an LSTM architecture for an NLP task, achieving mediocre results
- Why: For any NLP task with available pretrained transformers, fine-tuning BERT or a similar model will almost always outperform an LSTM trained from scratch, even with less data
- Fix: Default to transformer-based models for NLP. Only consider LSTMs for streaming/edge scenarios with strict memory constraints, or when processing truly continuous data streams

**Pitfall 4: Forgetting to initialize forget gate bias to 1.0+**
- Symptom: LSTM cannot learn long-range dependencies; performance is similar to a vanilla RNN despite using LSTM architecture
- Why: Default random initialization sets forget gate biases near zero, causing the sigmoid to output ~0.5, which halves the gradient at every timestep and effectively recreates the vanishing gradient problem
- Fix: After creating the LSTM, manually set the forget gate bias to 1.0 or 2.0. This biases the gate toward keeping information by default

## Hands-On Exercises

### Exercise 1: Vanilla RNN vs LSTM on Long Sequences
**Goal:** Empirically demonstrate the vanishing gradient problem and how LSTMs solve it
**Task:**
1. Create a synthetic "copy memory" task: the model sees a sequence of random digits followed by many zeros, then must reproduce the original digits at the end. Vary the gap (number of zeros) from 10 to 100.
2. Implement both a vanilla RNN and an LSTM in PyTorch with the same hidden size (128)
3. Train both on the task with gaps of 10, 20, 50, and 100 timesteps
4. Plot: (a) accuracy vs gap length for each architecture, (b) gradient norms at the first timestep during training
**Verify:** The vanilla RNN should succeed at gap=10 but fail at gap=50+. The LSTM (with forget gate bias=1.0) should succeed up to gap=100. The gradient norm plot should show the vanilla RNN's gradients vanishing to near-zero as gap length increases, while the LSTM's gradients remain stable.

### Exercise 2: Sentiment Classification with LSTM
**Goal:** Build a practical LSTM-based text classifier and compare with a simple baseline
**Task:**
1. Load the IMDB movie review dataset (25K train, 25K test)
2. Implement a bidirectional LSTM classifier: Embedding(vocab_size, 128) -> BiLSTM(128, hidden=256, num_layers=2, dropout=0.3) -> Linear(512, 1) -> Sigmoid
3. Use proper pack_padded_sequence for variable-length reviews
4. Train with Adam (lr=1e-3), gradient clipping (max_norm=1.0), and early stopping
5. Compare: (a) accuracy with and without bidirectional, (b) accuracy with and without pack_padded_sequence
**Verify:** The bidirectional LSTM should reach ~87-89% accuracy. Unidirectional should be ~2-3% lower. Without pack_padded_sequence, accuracy may drop by 1-2% depending on padding strategy.

---

## 10. Interview Questions

### Conceptual

1. **Explain the vanishing gradient problem in RNNs.** The same weight matrix is repeatedly multiplied during backpropagation through time. If its spectral radius is less than 1, gradients decay exponentially, preventing the network from learning dependencies beyond roughly 10-20 timesteps. The tanh activation makes this worse because its gradient is always less than or equal to 1.

2. **How do LSTMs solve the vanishing gradient problem?** LSTMs introduce a cell state with **additive** updates (no matrix multiplication). The gradient flows through the cell state by multiplying only by the forget gate value, which the network learns to keep near 1 when information needs to persist. This provides an unobstructed gradient highway.

3. **What is the difference between LSTM and GRU?** GRU merges the forget and input gates into a single update gate, and merges the cell state and hidden state into one vector. GRU has fewer parameters (3 vs 4 gate matrices), trains faster, and performs comparably on most tasks. LSTM has slightly better capacity for very long sequences due to the separate cell state.

4. **Why did transformers replace RNNs?** Three reasons: (1) Parallelization — RNNs process tokens sequentially while transformers process all tokens simultaneously, enabling 10-100x training speedups on GPUs. (2) Long-range dependencies — attention connects any two positions in O(1) path length vs O(T) for RNNs. (3) Scaling — transformers absorb more compute and data efficiently, which matters because scaling laws show reliable performance improvement with scale.

5. **What is teacher forcing and what problem does it create?** Teacher forcing feeds ground-truth previous tokens to the decoder during training, speeding convergence. But during inference, the model must use its own predictions, which may be wrong. This train-test mismatch (exposure bias) causes errors to compound during generation.

### System Design

6. **When would you use an RNN/LSTM over a transformer in 2026?** Real-time streaming on edge devices (constant memory, no KV cache growth), truly continuous time series (sensor data with no natural context window), embedded systems with kilobytes of RAM, and as a component in hybrid architectures. For any NLP task or any task with available pretrained transformers, use transformers.

7. **Design a system for real-time anomaly detection on IoT sensor streams.** Use a small GRU (hidden size 64) that processes one reading at a time. The hidden state captures normal patterns. When the reconstruction error (predicting the next reading) exceeds a learned threshold, flag an anomaly. The constant memory footprint fits on microcontrollers, and the sequential processing naturally handles streaming data.

---

## Key Takeaways

1. RNNs process sequences by maintaining a hidden state — a compressed summary updated at each timestep
2. Vanilla RNNs fail on long sequences because gradients vanish through repeated matrix multiplication
3. LSTMs solve this with a cell state that uses **additive** updates and learned gates (forget, input, output)
4. GRUs simplify LSTMs to two gates (update, reset) with comparable performance and fewer parameters
5. Bidirectional RNNs see both past and future context, but cannot be used for generation
6. The seq2seq encoder-decoder with attention was the direct precursor to transformers
7. Transformers replaced RNNs because of parallelization, O(1) long-range connections, and scaling efficiency
8. RNNs remain relevant for streaming/edge scenarios; state space models (Mamba) are their modern successor
9. If you are building anything in NLP in 2026, use a transformer unless you have a specific reason not to

## Summary

RNNs process sequences by maintaining a hidden state, but vanilla RNNs fail beyond ~20 timesteps due to vanishing gradients from repeated weight matrix multiplication. LSTMs solve this with a cell state that uses additive updates and learned gates, providing an unobstructed gradient highway. While transformers have replaced RNNs for nearly all NLP tasks (due to parallelization, O(1) long-range connections, and superior scaling), understanding RNNs is essential because their failure modes illuminate why transformers work, and the concept of maintaining compressed state is returning in state space models like Mamba.

## What's Next

- **Next lesson:** [Transformers and Attention](../transformers-attention/COURSE.md) -- the architecture that solved all the problems RNNs struggled with, using self-attention to connect any two positions in O(1) path length with full parallelization
- **Builds on this:** The attention mechanism in seq2seq models (Bahdanau, 2014) was the direct precursor to transformer self-attention -- the progression from RNN bottleneck to attention to full self-attention is the key historical arc of modern deep learning
