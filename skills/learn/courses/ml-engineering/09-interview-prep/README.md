# 09 — Interview Preparation

> ML engineering interviews: pair programming in your IDE, system design, and concept explanations. No hiding behind agents.

## Why This Matters

> "You will be expected to complete a pair programming interview, using your own IDE."
> "Our goal is to complete the entire interview loop within 30 days."

You need to demonstrate live knowledge. This section prepares for that.

## Subdirectories

```
09-interview-prep/
├── system-design/       # ML system design exercises (production-relevant scenarios)
├── concepts/            # 20 core concepts to explain without notes
└── pair-programming/    # Live coding practice scenarios
```

## System Design Exercises

Practice these timed (45 min each):

1. **AI Personal Shopper** — Design the full ML system for personalized product recommendations
2. **Product recommendation engine for 100M+ shoppers** — Scale, latency, cold start
3. **Real-time fraud detection for payment processing** — Streaming data, low latency, high recall
4. **Merchant churn prediction** — Tabular data, class imbalance, feature engineering
5. **Search ranking for e-commerce storefronts** — Learning to rank, embeddings, A/B testing

## 20 Concepts to Explain Without Notes

Practice explaining each in 2-3 minutes:

1. What is backpropagation and why does it work?
2. What is attention in transformers and why is it important?
3. What's the difference between fine-tuning and RAG? When would you use each?
4. What is overfitting and how do you prevent it?
5. Explain bias-variance tradeoff
6. What is gradient descent? Why does Adam work well?
7. What is LoRA and why is it useful?
8. What is RLHF and when would you use it?
9. How do you handle class imbalance in a dataset?
10. What is data drift and how do you detect it?
11. Explain precision vs recall. When do you optimize for each?
12. What is a confusion matrix and what does it tell you?
13. How would you deploy a model to production?
14. What is A/B testing for ML models?
15. What is a feature store and why would you use one?
16. Explain quantization — why and how?
17. What are embeddings and how are they used in recommendations?
18. What is the cold start problem in recommendations?
19. How does distributed training work (data parallelism)?
20. Your model has 95% accuracy but stakeholders say it's bad. What do you check?

## Pair Programming Practice

Topics to practice live coding:

- [ ] Build a simple data pipeline (load, clean, feature engineer, split)
- [ ] Train a model and evaluate with proper metrics
- [ ] Implement a basic fine-tuning script with LoRA
- [ ] Write a model serving endpoint
- [ ] Debug a model that isn't converging (intentionally broken)
