# Tokenization: How Models See Text

## Learning Objectives

By the end of this lesson, you will:

**Understand:**
- How subword tokenization algorithms (BPE, SentencePiece, WordPiece) construct vocabularies and encode text
- Why tokenization choices directly affect model behavior on numbers, code, and multilingual text
- The tradeoffs between vocabulary size, sequence length, and embedding parameter count

**Apply:**
- Inspect and compare tokenization outputs using tiktoken and HuggingFace tokenizers
- Estimate token counts, context window budgets, and API costs for production workloads

**Analyze:**
- Evaluate which tokenizer characteristics matter most for a given application domain (e.g., multilingual commerce vs. code generation)

## Prerequisites

- No strict prerequisites. This lesson is conceptual and can be read early in the curriculum. Familiarity with how neural networks consume numerical inputs is helpful but not required.

## Why Tokenization Matters

Neural networks operate on numbers, not text. Tokenization is the bridge — the process of converting text into a sequence of integers that a model can process, and converting model outputs back into text.

This sounds mundane, but tokenization decisions have profound downstream effects:
- **Context window efficiency**: A bad tokenizer wastes tokens on common words, reducing how much content fits in your context window.
- **Model behavior on numbers and code**: Tokenization artifacts explain why LLMs struggle with arithmetic and certain code patterns.
- **Multilingual performance**: A tokenizer trained primarily on English will use 2-5x more tokens for Chinese or Arabic text, making the model slower and more expensive for those languages.
- **Prompt engineering**: Understanding tokenization helps you write more efficient prompts.

If you're building an AI shopping assistant that handles product names, prices, and customer queries in multiple languages, tokenization directly affects your costs and quality.

## The Tokenization Spectrum

### Character-Level

The simplest approach: each character is a token.

```
"Hello world" → ['H', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd']
Vocabulary size: ~256 (ASCII) or ~150K (Unicode)
Tokens: 11
```

**Pros**: Tiny vocabulary, handles any text, no unknown tokens.
**Cons**: Sequences are very long (each character is a step). The model must learn to compose characters into words, words into sentences — this requires many more layers and more compute. A 4096-token context window would hold only ~4096 characters (~800 words).

Character-level models exist but are impractical for most LLM applications.

### Word-Level

Each word (whitespace-separated) is a token.

```
"Hello world" → ['Hello', 'world']
Vocabulary size: 100K-500K+ (every unique word)
Tokens: 2
```

**Pros**: Sequences are short. Each token carries a lot of meaning.
**Cons**: Huge vocabulary (memory-expensive), can't handle novel words or misspellings, doesn't generalize across languages with different word boundaries (Chinese has no spaces), morphological variations create separate tokens ("run", "running", "runs", "ran" are 4 unrelated tokens).

Word-level was used in early NLP (word2vec, GloVe) but is inadequate for modern LLMs.

### Subword-Level (The Standard)

The sweet spot: split text into units that are smaller than words but larger than characters. Common words stay whole, rare words get split into meaningful pieces.

```
"unhappiness" → ['un', 'happiness']      or ['un', 'happ', 'iness']
"tokenization" → ['token', 'ization']
"ChatGPT" → ['Chat', 'G', 'PT']
Vocabulary size: 32K-100K
```

Every modern LLM uses subword tokenization. The three main algorithms: BPE, SentencePiece, and WordPiece.

---

### Check Your Understanding

1. Why is character-level tokenization impractical for LLMs despite having no unknown tokens?
2. A word-level tokenizer encounters the word "unforgettable" for the first time at inference. What happens? How would a subword tokenizer handle it differently?
3. True or false: Subword tokenization guarantees that semantically related words (e.g., "run" and "running") share token components.

<details>
<summary>Answers</summary>

1. Character-level sequences are extremely long (each character is one token), meaning a 4096-token context window holds only about 800 words. The model must also learn to compose characters into meaningful units, requiring more layers and compute.
2. A word-level tokenizer would map it to an [UNK] (unknown) token, losing all information. A subword tokenizer would decompose it into known pieces, e.g., ["un", "forget", "table"] or ["un", "forgettable"], preserving meaning.
3. False. Subword tokenization is purely statistical (based on frequency), not semantic. "run" and "running" may or may not share subword tokens depending on corpus frequency. For example, "running" might be a single token if it appears frequently enough.

</details>

---

## BPE: Byte Pair Encoding

BPE is the most widely used tokenization algorithm. GPT-2, GPT-3, GPT-4, LLaMA, and most modern models use BPE or its variants.

### How BPE Training Works (Step by Step)

1. **Start with a base vocabulary** of all individual bytes (256 tokens for byte-level BPE) or characters.

2. **Count all adjacent pairs** in the training corpus.

3. **Merge the most frequent pair** into a new token. Add it to the vocabulary.

4. **Repeat** until you reach your desired vocabulary size.

Let's trace through a tiny example:

```
Corpus: "low low low low low lowest lowest newer newer newer wider wider"

Step 0 — Character vocabulary:
  {'l', 'o', 'w', 'e', 's', 't', 'n', 'r', 'i', 'd', ' '}

Step 1 — Most frequent pair: ('l', 'o') appears 7 times
  Merge: 'l' + 'o' → 'lo'
  Corpus becomes: "lo w lo w lo w lo w lo w lo w e s t lo w e s t ..."

Step 2 — Most frequent pair: ('lo', 'w') appears 7 times
  Merge: 'lo' + 'w' → 'low'

Step 3 — Most frequent pair: ('e', 'r') appears 5 times
  Merge: 'e' + 'r' → 'er'

Step 4 — Most frequent pair: ('n', 'ew') appears 3 times
  ...and so on until we reach target vocab size.
```

After training, common words like "the", "is", "and" become single tokens. Rare words get decomposed into subword units. The merge rules are saved and applied during encoding.

### BPE Encoding (At Inference)

To tokenize new text, apply the learned merge rules greedily:

```
"lowest" → start with characters: ['l', 'o', 'w', 'e', 's', 't']
  Apply merge 'l'+'o' → 'lo':   ['lo', 'w', 'e', 's', 't']
  Apply merge 'lo'+'w' → 'low': ['low', 'e', 's', 't']
  Apply merge 'e'+'s' → 'es':   ['low', 'es', 't']
  Apply merge 'es'+'t' → 'est': ['low', 'est']
  Final tokens: ['low', 'est']
```

### Byte-Level BPE

GPT-2 introduced a crucial modification: start from bytes (256 base tokens) instead of Unicode characters. This means:
- **No unknown tokens**: Any text can be encoded (it falls back to individual bytes)
- **Language agnostic**: Works on any language, any script
- **Handles binary/special characters**: Emojis, code, mathematical symbols — everything

The downside: sequences can be longer for non-Latin scripts, since multi-byte Unicode characters may be split into individual bytes.

## SentencePiece

SentencePiece (Kudo & Richardson, 2018) is a tokenization framework that treats text as a raw byte stream — no language-specific preprocessing (no whitespace splitting, no lowercasing).

### Key Properties

- **Language agnostic**: Doesn't assume spaces separate words. Critical for Chinese, Japanese, Thai, and other languages without explicit word boundaries.
- **Reversible**: Can perfectly reconstruct the original text from tokens (including whitespace).
- **Treats whitespace as a character**: Spaces become the Unicode meta symbol and are part of tokens. This is why you see things like "The" with a leading marker in SentencePiece vocabularies — it encodes the space before the word.
- **Supports BPE and Unigram**: SentencePiece is a framework, not an algorithm. It can use either BPE or Unigram LM internally.

```
SentencePiece BPE:
"Hello, world!" → ['_Hello', ',', '_world', '!']

Note the _ prefix on word-initial tokens — this encodes the space before the word.
```

LLaMA, Mistral, and many multilingual models use SentencePiece.

### Unigram Model (SentencePiece variant)

Instead of building up a vocabulary by merging (BPE), Unigram starts with a large vocabulary and prunes it down:

1. Start with a large candidate vocabulary (all substrings up to a length, or BPE-generated candidates)
2. Assign probabilities to each token using EM (Expectation-Maximization)
3. Remove tokens that contribute least to the corpus likelihood
4. Repeat until reaching target vocabulary size

Unigram can find more globally optimal tokenizations than BPE's greedy approach.

## WordPiece

WordPiece is BERT's tokenizer. Similar to BPE but with a key difference in the merge criterion:

- **BPE**: Merge the most *frequent* pair
- **WordPiece**: Merge the pair that maximizes the *likelihood of the training corpus*

```
WordPiece:
"unhappiness" → ['un', '##happiness']  or ['un', '##hap', '##piness']

The ## prefix indicates a continuation token (not a word start).
```

WordPiece is used by BERT, DistilBERT, and other encoder models. For decoder-only LLMs (GPT, LLaMA), BPE/SentencePiece dominate.

---

### Check Your Understanding

1. In BPE, what determines which pair of tokens gets merged at each step? How does WordPiece differ?
2. Why does byte-level BPE guarantee there are no unknown tokens?
3. SentencePiece treats whitespace as a regular character using the Unicode meta symbol. Why is this important for languages like Chinese or Thai?

<details>
<summary>Answers</summary>

1. BPE merges the most frequent adjacent pair. WordPiece merges the pair that maximizes the likelihood of the training corpus, which considers not just frequency but how much the merge improves the overall model of the data.
2. Byte-level BPE starts with a base vocabulary of all 256 byte values. Since any text can be represented as a sequence of bytes, the tokenizer can always fall back to individual bytes for unknown strings. No input is unrepresentable.
3. These languages do not use spaces to separate words. SentencePiece treats the raw byte stream without assuming whitespace-delimited words, so it can learn appropriate subword units for any language without language-specific preprocessing.

</details>

---

## Vocabulary Size Tradeoffs

| Vocab Size | Tokens per Word | Embedding Params | Coverage | Model |
|-----------|-----------------|------------------|----------|-------|
| 32K | ~1.3-1.5 | 32K * d_model | Good for English | LLaMA 1/2 |
| 50K | ~1.2-1.4 | 50K * d_model | Better multilingual | GPT-2 |
| 100K | ~1.1-1.3 | 100K * d_model | Excellent coverage | GPT-4, LLaMA 3 |
| 256K | ~1.0-1.2 | 256K * d_model | Near word-level | Gemini |

### The Tradeoff

**Larger vocabulary:**
- Fewer tokens per text, meaning shorter sequences, faster inference, more context
- Better coverage of rare words and multilingual text
- More embedding parameters (vocabulary_size * hidden_dim)
- Each token is more "specific" — less ambiguity

**Smaller vocabulary:**
- More tokens per text, meaning longer sequences, slower, less context
- More word splitting and the model must learn to compose subwords
- Fewer embedding parameters
- More robust to rare/novel words (falls back to character-level)

**The trend**: Vocabulary sizes have been increasing. LLaMA 3 moved from 32K to 128K tokens, citing significant efficiency gains especially for code and multilingual text.

### Embedding Table Memory

For a 128K vocabulary with d_model=4096:
```
Embedding params: 128,000 * 4,096 = 524M parameters
In bf16: 524M * 2 bytes = ~1 GB
```

For a 7B model, that's ~7% of total parameters just for the embedding table. Non-trivial but manageable.

## Special Tokens

Every tokenizer defines special tokens with specific meanings:

```
<|endoftext|>  — End of document (GPT-2/3)
<s>, </s>      — Beginning/end of sequence (LLaMA)
[CLS]          — Classification token (BERT)
[SEP]          — Separator between segments (BERT)
[PAD]          — Padding token (for batching)
[MASK]         — Masked token (BERT MLM training)
[UNK]          — Unknown token (rare with BPE)

Chat-specific:
<|im_start|>, <|im_end|>  — ChatML format markers
<|system|>, <|user|>, <|assistant|>  — Role markers
```

Special tokens are not split by the tokenizer — they're always single tokens regardless of their string content. They're added to the vocabulary explicitly and given specific IDs.

**Important for prompt engineering**: Special tokens affect how the model interprets your input. Misusing them (or accidentally including them in user input) can cause unexpected behavior. Always sanitize user input to escape or remove special token strings.

## Tokenization Gotchas

### Numbers

Most tokenizers are terrible with numbers because they tokenize digits inconsistently:

```
"123456" might tokenize as:
  ['123', '456']      — treated as two separate tokens
  ['12', '34', '56']  — three tokens
  ['1', '23', '4', '56']  — four tokens, inconsistent grouping
```

The model never sees "123456" as a single number. It sees arbitrary chunks. This is a fundamental reason LLMs struggle with arithmetic — they don't have a consistent representation of numbers.

**Workarounds**: Some newer tokenizers add individual digit tokens. Some approaches encode each digit separately ("1 2 3 4 5 6"). Research on number-aware tokenization is ongoing.

### Code

Code tokenization has unique challenges:

```python
# Indentation is meaningful in Python
"    def hello():"
→ ['    ', 'def', ' hello', '():', '\n']  # 4 spaces = 1 token? or 4 tokens?
```

Different tokenizers handle whitespace differently. GPT-4's tokenizer was specifically designed with code in mind — it handles indentation patterns efficiently. LLaMA's original 32K tokenizer was less code-friendly (one reason they expanded to 128K).

### Multilingual Text

A tokenizer trained primarily on English will be inefficient for other languages:

```
English: "Hello, how are you?" → 6 tokens
Chinese: "你好，你怎么样？"    → 12-18 tokens (depending on tokenizer)
Thai:    "สวัสดีครับ คุณสบายดีไหม" → 20-30 tokens
```

The same semantic content uses 2-5x more tokens in non-English languages. This means:
- **Higher costs** for non-English users (more tokens = higher API bills)
- **Less context** available (the context window fills up faster)
- **Potentially worse quality** (model has seen fewer tokens of non-English text per concept)

LLaMA 3's expanded vocabulary specifically addressed this, adding tokens for common Chinese, Japanese, Korean, and other language patterns.

### Trailing Whitespace and Formatting

```
"Hello " is not equal to "Hello"   — the space changes the tokenization
"Hello\n" is not equal to "Hello"  — newlines matter
```

This affects prompt engineering: adding or removing a trailing space can change which token the model generates next, sometimes significantly.

---

### Check Your Understanding

1. A 7B model uses a 128K vocabulary with d_model=4096. How many parameters are in the embedding table alone, and how much memory does that consume in bf16?
2. You are choosing between a 32K and 100K vocabulary for a multilingual model. What are two arguments for the larger vocabulary?
3. Why has the trend in recent models been toward larger vocabularies?

<details>
<summary>Answers</summary>

1. 128,000 x 4,096 = 524 million parameters. In bf16 (2 bytes per parameter): 524M x 2 = ~1 GB. This is roughly 7% of a 7B model's total parameters.
2. (a) Better coverage of non-English tokens, reducing the token count for multilingual text and lowering costs. (b) Fewer tokens per sequence, meaning faster inference and more content fitting in the context window.
3. The efficiency gains are significant: fewer tokens per input means faster inference, lower cost, and more effective use of the context window, especially for code and multilingual text. The memory cost of larger embedding tables is manageable relative to total model size.

</details>

---

## How Tokenization Affects Model Behavior

### Glitch Tokens

Some tokens in the vocabulary are "undertrained" — they appeared in the tokenizer training data (for merge purposes) but rarely in the model's training data. These tokens can cause bizarre behavior:

```
The famous GPT-3 " SolidGoldMagikarp" token:
- A valid BPE token (from a Reddit username)
- Barely appeared in training data
- Asking the model about it caused hallucination and incoherent outputs
- The embedding for this token was essentially random noise
```

Glitch tokens are a consequence of the tokenizer and model being trained on different data distributions, or from tokens that are technically valid BPE merges but represent rare strings.

### Arithmetic Failures

```
"What is 37 + 28?"

Tokenization: ['What', ' is', ' 37', ' +', ' 28', '?']

The model sees "37" as a single opaque token, not as "3" and "7".
It must learn addition as a lookup table over token pairs, not as an algorithm.
For multi-digit numbers with inconsistent tokenization, this breaks down.
```

This is why chain-of-thought prompting helps with math — it forces the model to work through intermediate steps, reducing the arithmetic complexity per step.

### Spelling and Character-Level Tasks

```
"How many r's in 'strawberry'?"

The model sees: ['str', 'aw', 'berry'] (or similar)
It never sees individual characters unless they happen to be single-character tokens.
So counting letters requires reasoning about subword composition — something models
aren't explicitly trained to do.
```

## Practical: Inspecting Tokenization

Every ML engineer should be comfortable inspecting how their text gets tokenized.

### Using Tiktoken (OpenAI's tokenizer)

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")  # GPT-4's tokenizer

text = "Hello, world! Price: $29.99"
tokens = enc.encode(text)
print(f"Token count: {len(tokens)}")
print(f"Tokens: {tokens}")
print(f"Decoded: {[enc.decode([t]) for t in tokens]}")

# Output:
# Token count: 10
# Tokens: [9906, 11, 1917, 0, 8650, 25, 400, 1682, 13, 2079]
# Decoded: ['Hello', ',', ' world', '!', ' Price', ':', ' $', '29', '.', '99']
```

### Using HuggingFace Tokenizers

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

text = "Hello, world! Price: $29.99"
tokens = tokenizer.encode(text)
decoded = tokenizer.convert_ids_to_tokens(tokens)
print(f"Tokens: {decoded}")
print(f"Count: {len(tokens)}")
```

### Online Tools

- **Tiktokenizer** (tiktokenizer.vercel.app): Visual tokenizer for OpenAI models
- **HuggingFace Tokenizer playground**: Test any HF tokenizer
- Most model API dashboards show token counts

### What to Look For

When inspecting tokenization for your use case:

1. **Token count for typical inputs**: How many tokens does a typical product description use? A customer query?
2. **Price and number handling**: How are "$29.99", "SKU-12345", phone numbers tokenized?
3. **Multilingual efficiency**: If you serve global customers, how many tokens does the same question use in different languages?
4. **Special characters**: Product names with trademarks, emojis in reviews, URL patterns.
5. **Context budget**: System prompt + RAG context + conversation history — how many tokens total? Does it fit in your context window?

## Why This Matters for Prompt Engineering

### Token Efficiency

Every token counts toward your context window and your API bill.

```
Inefficient prompt (82 tokens):
"I would like you to please help me by providing a detailed and comprehensive
 summary of the following product review, making sure to capture all the key
 points and sentiment expressed by the customer."

Efficient prompt (15 tokens):
"Summarize this product review, noting key points and sentiment:"
```

Same semantic content, ~5x fewer tokens. At scale, this matters.

### Token Boundaries Affect Generation

The model generates one token at a time. Token boundaries affect what the model "sees" as atomic units:

```
Prompt ending: "The price is $"
Next token prediction: "29" (common price token)
Then: ".99"

vs.

Prompt ending: "The price is $2"
Next token prediction: "9" (digit continuation)
Then: ".99" or ".95" etc.
```

Where you split your prompt relative to token boundaries can influence generation. This is subtle but real.

### Chat Template Overhead

Every message in a multi-turn conversation includes template tokens:

```
<|im_start|>user\n       — 3-4 tokens of overhead
<|im_end|>\n              — 2 tokens of overhead
<|im_start|>assistant\n   — 3-4 tokens of overhead
```

A 20-turn conversation can spend 150+ tokens just on template formatting. Factor this into context window budgets.

## Common Pitfalls

1. **Assuming tokens are words.** Many practitioners count words and assume that is the token count. In practice, one word can be multiple tokens and one token can span parts of multiple words. Always use the actual tokenizer to get token counts.
2. **Ignoring chat template overhead.** System prompts, role markers, and turn delimiters consume tokens that are invisible to the user but eat into the context window. A 20-turn conversation can spend 150+ tokens on template formatting alone.
3. **Forgetting that tokenizers are fixed after training.** The tokenizer is frozen before model training begins. You cannot change it after the fact without retraining the model. If you need a different tokenizer (e.g., for better multilingual support), you need a different model or must retrain.
4. **Testing prompts without inspecting tokenization.** Subtle changes like trailing whitespace, newlines, or different number formatting can shift token boundaries and change model behavior. Always inspect tokenization when debugging unexpected outputs.

## Hands-On Exercises

### Exercise 1: Compare Tokenizers Across Languages (15 min)

Using Python, compare how two different tokenizers handle the same sentence in English, Chinese, and Spanish.

```python
# Install: pip install tiktoken transformers
import tiktoken
from transformers import AutoTokenizer

sentences = {
    "English": "What running shoes do you recommend under $100?",
    "Chinese": "你推荐哪些100美元以下的跑步鞋？",
    "Spanish": "Que zapatos para correr recomiendas por menos de $100?",
}

# Compare: tiktoken cl100k_base (GPT-4) vs. a LLaMA 3 tokenizer
# For each sentence, print the token count and the decoded tokens.
# Question: Which tokenizer is more efficient for non-English text? By how much?
```

### Exercise 2: Tokenization and Arithmetic (20 min)

Investigate how GPT-4's tokenizer handles numbers of different lengths and formats. Tokenize the following and observe the patterns:
- "123", "1234", "12345", "123456", "1234567"
- "$29.99", "$299.99", "$2999.99"
- "3 + 7 = 10", "37 + 28 = 65", "370 + 280 = 650"

Questions to answer:
1. At what number length does tokenization become inconsistent (different digit groupings)?
2. How does the "$" sign interact with the number tokens?
3. Based on your observations, explain why LLMs struggle more with multi-digit arithmetic than single-digit arithmetic.

## Interview Questions

**Conceptual:**
1. Explain how BPE tokenization works, step by step. How is the vocabulary built?
2. What's the difference between BPE, SentencePiece, and WordPiece? When would you use each?
3. Why do LLMs struggle with arithmetic? How does tokenization contribute to this problem?
4. What are the tradeoffs of vocabulary size? Why has the trend been toward larger vocabularies?
5. What is a "glitch token" and why does it happen?

**Applied:**
6. You're building a multilingual shopping assistant. The tokenizer was trained mostly on English text. What problems will you encounter and how would you mitigate them?
7. A product catalog has many entries like "SKU-A1B2C3-XL-BLUE". How would current tokenizers handle this? Does it matter for your application?
8. Your model is hitting the context window limit during multi-turn shopping conversations. Beyond switching to a model with a larger context, what can you do?
9. You notice that your model gives inconsistent results when prices are formatted differently ("$29.99" vs "29.99 USD" vs "$29,99"). Explain why this happens at the tokenization level.
10. How would you estimate the monthly API cost for a shopping assistant handling 1M conversations per month, averaging 10 turns each?

**Answer to Q10**: Estimate tokens per turn — user message ~50 tokens, assistant response ~150 tokens, context overhead ~50 tokens. Per turn: ~250 tokens processed. Per conversation (10 turns, but context grows): roughly sum of 250, 500, 750, ..., 2500 = ~13,750 total tokens processed. At $0.01/1K input + $0.03/1K output tokens (GPT-4o-mini pricing): ~$0.005 per conversation. At 1M conversations: ~$5,000/month. With a larger model at 10x pricing: ~$50,000/month. This is why model routing and caching matter enormously.

## Summary

This lesson covered how text is converted to numerical tokens for LLM consumption. The key takeaways:

- **Subword tokenization** (BPE, SentencePiece, WordPiece) is the standard, balancing vocabulary size against sequence length.
- **BPE** builds vocabulary bottom-up by iteratively merging the most frequent pairs. Byte-level BPE eliminates unknown tokens entirely.
- **Vocabulary size** involves a direct tradeoff: larger vocabularies produce shorter sequences (faster, cheaper) but require more embedding parameters.
- **Tokenization artifacts** explain many LLM failure modes: inconsistent number splitting causes arithmetic errors, subword boundaries cause spelling task failures, and undertrained tokens cause glitch behavior.
- **Multilingual efficiency** varies dramatically by tokenizer -- the same content can cost 2-5x more tokens in non-English languages.
- **Practical inspection** of tokenization is essential for prompt engineering, cost estimation, and debugging model behavior.

## What's Next

The next lesson, **Pretraining** (see [Pretraining](../pretraining/COURSE.md)), covers how models are trained from scratch on massive text corpora using the causal language modeling objective. Understanding tokenization is foundational to pretraining, since the tokenizer determines what "tokens" the model actually predicts.
