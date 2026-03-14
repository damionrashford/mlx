# Learning Path — AI-Native ML Engineer (8 Months)

> Approach: Learn the WHY, WHEN, and WHAT. Agents write code. You make decisions.
> Start date: 2026-03-08
> Target: Apply for Applied ML Engineering roles Q4 2026

## Phase 1: Build the Intuition Layer (Month 1-2)

**Goal:** Understand how neural networks and transformers actually work — the concepts, not the code.

### Week 1-2: Visual Math Foundations
- [ ] 3Blue1Brown: Essence of Linear Algebra (16 videos, ~3.5 hrs)
- [ ] 3Blue1Brown: Essence of Calculus (12 videos, ~3 hrs)
- [ ] 3Blue1Brown: Neural Networks (4 videos, ~1 hr)
- [ ] 3Blue1Brown: Backpropagation (~20 min)
- [ ] StatQuest: Gradient Descent (~15 min)
- [ ] StatQuest: Cross Entropy (~15 min)
- [ ] StatQuest: Bias vs Variance (~20 min)
- [ ] **Practice:** Explain each concept out loud without looking at notes

### Week 3-4: Transformer & LLM Intuition
- [ ] DeepLearning.AI: How Transformer LLMs Work (~2 hrs)
- [ ] DeepLearning.AI: Attention in Transformers — Concepts & Code in PyTorch (~2 hrs)
- [ ] Read: Jay Alammar — "The Illustrated Transformer" blog
- [ ] Read: Jay Alammar — "The Illustrated GPT-2" blog
- [ ] Read: Andrej Karpathy — "The Unreasonable Effectiveness of RNNs" blog
- [ ] **Practice:** Open Claude Code, ask it to build a transformer. Evaluate every component.

### Week 5-6: LLM Vocabulary & Technique Landscape
- [ ] Build personal glossary of 50+ ML/LLM terms (see `references/decision-frameworks/`)
- [ ] DeepLearning.AI: Finetuning Large Language Models (~2 hrs)
- [ ] DeepLearning.AI: Pretraining LLMs (~2 hrs)
- [ ] DeepLearning.AI: Quantization Fundamentals (~2 hrs)
- [ ] **Practice:** For each technique (LoRA, RLHF, RAG, distillation, etc.) write a one-paragraph "when would I use this?"

### Week 7-8: Fine-tuning & Alignment
- [ ] DeepLearning.AI: Fine-tuning & RL for LLMs — Intro to Post-Training (~2 hrs)
- [ ] DeepLearning.AI: Reinforcement Fine-Tuning LLMs — GRPO (~2 hrs)
- [ ] DeepLearning.AI: Build and Train an LLM with JAX (~2 hrs)
- [ ] **Practice:** Fine-tune a small model (Llama 3.2 1B) on a dataset using Claude Code — YOU make every design decision

---

## Phase 2: Understand the Full ML Lifecycle (Month 3-4)

**Goal:** Know the full pipeline from data to deployment. Understand WHEN to use each approach.

### Week 9-12: Generative AI with LLMs (Course)
- [ ] DeepLearning.AI: Generative AI with LLMs (~16 hrs)
  - Pretraining lifecycle
  - Fine-tuning strategies
  - RLHF and alignment
  - Deployment considerations
- [ ] Write notes in `03-llm-internals/` for each module

### Week 13-14: Classical ML Breadth
- [ ] DeepLearning.AI: Machine Learning Specialization — audit conceptual lectures
  - Focus on Andrew Ng's intuition, skip coding exercises (agents can do those)
  - Supervised learning: regression, classification, SVMs, trees, ensembles
  - Unsupervised: K-means, PCA, anomaly detection
  - Recommender systems (directly relevant to e-commerce ML roles)
- [ ] Write decision framework: "which model for which problem?" in `references/decision-frameworks/`

### Week 15-16: Evaluation & Debugging
- [ ] DeepLearning.AI: Improving Accuracy of LLM Applications (~2 hrs)
- [ ] DeepLearning.AI: Evaluating & Debugging Generative AI with W&B (~2 hrs)
- [ ] Build ML debugging decision tree in `references/decision-frameworks/`
- [ ] **Practice:** Intentionally break a model (wrong LR, data leak, class imbalance) and diagnose it

---

## Phase 3: Production & System Design (Month 5-6)

**Goal:** Know how to take a model from notebook to production. Design ML systems.

### Week 17-20: Production ML
- [ ] DeepLearning.AI: Machine Learning in Production (~16 hrs)
  - ML lifecycle and scoping
  - Deployment patterns (edge vs cloud)
  - Data pipelines and quality
  - Monitoring and drift detection
- [ ] DeepLearning.AI: Efficiently Serving LLMs (~2 hrs)
- [ ] DeepLearning.AI: LLMOps (~2 hrs)
- [ ] Write notes in `05-production-ml/` for each topic

### Week 21-22: Data Engineering
- [ ] Google BigQuery free tier — hands-on practice
- [ ] DBT tutorials (docs.getdbt.com) — build a sample transform
- [ ] DeepLearning.AI: Data Engineering course (audit)
- [ ] Write notes in `06-data-engineering/`

### Week 23-24: System Design Practice
- [ ] Design exercise: "ML system for an AI Personal Shopper"
  - Architecture diagram
  - Data pipeline design
  - Model selection and justification
  - Monitoring and evaluation plan
  - Write up in `09-interview-prep/system-design/`
- [ ] Design exercise: "Product recommendation engine for 100M+ shoppers"
- [ ] Design exercise: "Real-time fraud detection for a payments platform"

---

## Phase 4: Agent Edge + Interview Prep (Month 7-8)

**Goal:** Combine ML + agents into your unique value prop. Prepare for interviews.

### Week 25-26: Agent-ML Integration
- [ ] DeepLearning.AI: Evaluating AI Agents (~2 hrs)
- [ ] DeepLearning.AI: DSPy — Build Optimize Agentic Apps (~2 hrs)
- [ ] DeepLearning.AI: Building and Evaluating Data Agents (~2 hrs)
- [ ] DeepLearning.AI: Semantic Caching for AI Agents (~2 hrs)

### Week 27-28: Portfolio Project
- [ ] Build capstone: Train a product recommendation model → deploy as agent tool
  - Data collection pipeline
  - Model training (fine-tuned or classical)
  - Serve via API
  - Agent that calls the model as a tool
  - Write up in `10-portfolio-projects/product-recommender/`

### Week 29-30: Interview Prep
- [ ] Practice explaining 20 core concepts out loud (see `09-interview-prep/concepts/`)
- [ ] Mock system design interviews (3 designs, timed)
- [ ] Pair programming practice: build ML pipelines live in IDE
- [ ] Review all decision frameworks

### Week 31-32: Apply
- [ ] Update LinkedIn with new skills and projects
- [ ] Polish GitHub portfolio
- [ ] Apply for Applied ML Engineering roles
- [ ] Prepare for 30-day interview loop

---

## Progress Key

- [ ] Not started
- [~] In progress
- [x] Completed
