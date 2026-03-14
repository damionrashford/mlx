## Learning Objectives

By the end of this lesson, you will:

**Understand:**
- How Bayes' theorem connects priors, likelihoods, and posteriors, and why this framework underpins classification, regularization, and Bayesian inference
- Why standard loss functions (MSE, cross-entropy) are not arbitrary choices but derive directly from Maximum Likelihood Estimation under specific distributional assumptions
- What entropy, cross-entropy, and KL divergence measure, and why cross-entropy is the correct loss for classification

**Apply:**
- Derive the connection between a distributional assumption (Gaussian, Bernoulli, categorical) and its corresponding loss function via MLE
- Evaluate model comparisons using hypothesis testing (paired t-test, bootstrap confidence intervals) rather than naive single-number comparisons

**Analyze:**
- Assess when a model's reported accuracy is misleading due to class imbalance, and select appropriate evaluation metrics (precision, recall, F1, AUROC, PR-AUC)

## Prerequisites

- **Calculus** — integrals and derivatives are needed for continuous probability distributions, expectations, and understanding how MLE derives loss functions (see [Calculus](../calculus/COURSE.md))

---

# Probability and Statistics for Machine Learning

## Why This Matters

Machine learning IS applied statistics. A classifier outputs probabilities. A generative model samples from learned distributions. Training a model maximizes likelihood. Evaluating a model requires hypothesis testing. Loss functions are derived from probabilistic principles.

When someone says "the model is 92% confident this is a cat," they're making a probabilistic statement. When you choose cross-entropy over MSE for classification, that choice comes from probability theory. When you do A/B testing to compare two model architectures, that's hypothesis testing.

You don't need to prove theorems. You need to think probabilistically — to understand what your model is doing in terms of distributions, likelihoods, and uncertainty.

---

## 1. Probability Basics

### The Setup

A probability `P(A)` is a number between 0 and 1 representing how likely event A is. That's the frequentist view: "If I repeated this experiment infinitely many times, what fraction of the time would A happen?"

The Bayesian view is different: probability represents your *degree of belief*. `P(A) = 0.7` means "I'm 70% confident A is true." This is the view that dominates modern ML.

### Joint, Marginal, and Conditional Probability

**Joint probability** `P(A, B)`: probability that BOTH A and B happen.

**Marginal probability** `P(A)`: probability of A regardless of B. You get it by summing over all possible B values: `P(A) = sum_B P(A, B)`.

**Conditional probability** `P(A | B)`: probability of A *given that* B happened.

```
P(A | B) = P(A, B) / P(B)
```

ML example: "What's the probability of 'positive sentiment' given this particular sentence?" That's `P(positive | sentence)` — and it's exactly what a sentiment classifier outputs.

### Independence

Events A and B are independent if knowing B tells you nothing about A:

```
P(A | B) = P(A)
P(A, B) = P(A) * P(B)
```

**Why this matters in ML**: Naive Bayes assumes features are independent given the class. This assumption is almost always wrong (word frequencies ARE correlated), yet Naive Bayes often works well. Understanding why requires understanding conditional independence and its practical implications.

### Bayes' Theorem

```
P(A | B) = P(B | A) * P(A) / P(B)

Or in ML terms:
P(hypothesis | data) = P(data | hypothesis) * P(hypothesis) / P(data)
  posterior           =    likelihood         *    prior       / evidence
```

This is the foundation of Bayesian machine learning. You start with a prior belief about your model parameters, observe data, and update to a posterior belief.

**Concrete example**: Spam detection.

```
P(spam | "buy viagra now") = P("buy viagra now" | spam) * P(spam) / P("buy viagra now")

- P(spam) = 0.3  (30% of emails are spam — your prior)
- P("buy viagra now" | spam) = 0.01  (1% of spam contains this exact phrase)
- P("buy viagra now") = 0.0031  (overall frequency)

P(spam | "buy viagra now") = 0.01 * 0.3 / 0.0031 ≈ 0.97
```

97% probability it's spam, given the phrase. Bayes theorem turned `P(phrase | spam)` (easy to estimate from training data) into `P(spam | phrase)` (what we actually want).

### Interview-Ready Explanation

> "Bayes' theorem lets us flip conditional probabilities — we can compute `P(class | data)` from `P(data | class)` and our prior beliefs. This is foundational to ML: classifiers are estimating posterior probabilities, and many models can be interpreted as maximum a posteriori (MAP) estimation. The prior term is where regularization comes from — L2 regularization corresponds to a Gaussian prior on weights."

---

## 2. Probability Distributions

### Why Distributions Matter

A distribution tells you all possible outcomes and their probabilities. When a model "outputs a probability distribution," it's telling you: for each possible outcome, here's how likely I think it is.

### Bernoulli Distribution

Binary outcome: success (1) with probability `p`, failure (0) with probability `1-p`.

```
P(X = 1) = p
P(X = 0) = 1 - p
```

**ML context**: Binary classification. A sigmoid output gives you `p` — the probability of the positive class. The loss function (binary cross-entropy) is derived directly from the Bernoulli likelihood.

### Categorical Distribution

Generalization of Bernoulli to `k` classes. Each class has probability `p_i`, and `sum(p_i) = 1`.

**ML context**: Multi-class classification. Softmax output gives you the categorical distribution over classes. The loss (categorical cross-entropy) comes from the categorical likelihood.

### Normal (Gaussian) Distribution

The bell curve. Defined by mean `mu` and standard deviation `sigma`.

```
P(x) = (1 / (sigma * sqrt(2*pi))) * exp(-(x - mu)^2 / (2 * sigma^2))
```

**Why it's everywhere:**
- **Central Limit Theorem**: The average of many independent random variables is approximately normal, regardless of the underlying distribution. This is why many real-world quantities are roughly normal.
- **Weight initialization**: We initialize weights from `N(0, 1/sqrt(n))` or similar — a Gaussian.
- **Gaussian noise**: Added as regularization, or modeled in VAEs.
- **Linear regression**: Assumes errors are normally distributed. Minimizing MSE = maximizing Gaussian likelihood.
- **Batch norm**: Normalizes activations to have roughly standard normal distribution.

### Why Softmax Outputs a Distribution

Softmax converts raw logits (any real numbers) into a proper probability distribution:

```python
def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))  # subtract max for stability
    return exp_logits / exp_logits.sum()

# Example:
logits = [2.0, 1.0, 0.1]
softmax(logits)  # [0.659, 0.242, 0.099] — sums to 1.0
```

Key properties:
- All outputs are positive (exponential is always positive)
- All outputs sum to 1 (we divide by the sum)
- Preserves ranking (highest logit gets highest probability)
- Temperature parameter scales the "confidence": `softmax(logits / T)`. Low T = more confident. High T = more uniform.

**Temperature scaling** is used in LLMs: lower temperature for factual answers (more peaked distribution), higher temperature for creative writing (more uniform, more "surprising" token choices).

### Interview-Ready Explanation

> "Softmax converts raw neural network outputs (logits) into a valid probability distribution — all positive, summing to one. It's the natural link between a neural network's linear output and categorical probabilities. The exponential amplifies differences between logits, and the temperature parameter controls this amplification. This is why cross-entropy is the right loss for classification: it measures the distance between the predicted distribution (softmax output) and the true distribution (one-hot target)."

---

### Check Your Understanding

1. A softmax classifier outputs `[0.7, 0.2, 0.1]` for three classes. You lower the temperature from 1.0 to 0.5 (computing `softmax(logits / 0.5)`). What happens to the distribution, and when would you want this behavior?
2. Why does Naive Bayes work reasonably well in practice despite its assumption that features are conditionally independent (which is almost always violated)?
3. A model for medical diagnosis outputs `P(disease | symptoms) = 0.95`. The disease has a base rate of 0.001 in the population. A colleague says "the model is 95% accurate." Using Bayes' theorem reasoning, explain why this might be misleading.

<details>
<summary>Answers</summary>

1. Lowering the temperature makes the distribution more peaked — the highest probability (0.7) becomes even higher, and the others shrink closer to zero. This produces more "confident" predictions. You want low temperature when you need decisive outputs (e.g., factual question answering in LLMs). High temperature produces more uniform distributions, useful for creative generation or exploration.
2. Naive Bayes works because the independence assumption, while wrong, often does not change the *ranking* of class probabilities. The model may produce poorly calibrated probability values, but the most probable class according to Naive Bayes is often the correct one. Additionally, when training data is limited, the simpler model (fewer parameters to estimate) can outperform more complex models that overfit.
3. P(disease | symptoms) = 0.95 sounds impressive, but consider the reverse: even with a good test, if the base rate is 0.001, most positive results may be false positives. Using Bayes' theorem: `P(disease | positive test) = P(positive | disease) * P(disease) / P(positive)`. With a rare disease, P(positive) includes many false positives from the healthy population. The 95% likely refers to the model's output confidence, not the true posterior probability accounting for disease prevalence. Calibration matters.

</details>

---

## 3. Maximum Likelihood Estimation (MLE)

### The Core Idea

Given data, which model parameters make the observed data most probable?

```
theta_MLE = argmax_theta P(data | theta)
```

"Find the parameters theta that maximize the probability of having seen this data."

### MLE Derives Your Loss Functions

This is the key insight: **standard loss functions are not arbitrary choices. They come from MLE under specific distributional assumptions.**

**Linear regression (MSE loss):**
- Assume errors are Gaussian: `y = Wx + b + epsilon`, where `epsilon ~ N(0, sigma^2)`
- The likelihood is: `P(y | x, W, b) = Normal(Wx + b, sigma^2)`
- Taking log and maximizing → minimizing `sum((y_pred - y_true)^2)` → that's MSE

**Binary classification (binary cross-entropy):**
- Assume the label is Bernoulli: `y ~ Bernoulli(sigmoid(Wx + b))`
- The log-likelihood is: `y * log(p) + (1-y) * log(1-p)` — that's binary cross-entropy

**Multi-class classification (categorical cross-entropy):**
- Assume the label is categorical: `y ~ Categorical(softmax(Wx + b))`
- The log-likelihood is: `sum(y_true * log(y_pred))` — that's categorical cross-entropy

So when someone asks "why do we use cross-entropy for classification and MSE for regression?" the answer is: "Because those are the loss functions derived from MLE under the distributional assumptions appropriate for each task."

### MAP: MLE Plus a Prior

Maximum A Posteriori estimation adds a prior:

```
theta_MAP = argmax_theta P(theta | data) = argmax_theta P(data | theta) * P(theta)
```

If you assume a Gaussian prior on weights (`P(theta) ~ N(0, sigma^2)`), the log-prior becomes `-lambda * ||theta||^2`. Adding this to the log-likelihood gives you **L2 regularization**.

If you assume a Laplace prior, you get **L1 regularization**.

Regularization isn't a hack — it's Bayesian inference with a prior belief that weights should be small.

### Interview-Ready Explanation

> "MLE finds model parameters that maximize the probability of the observed data. It's not just a theoretical concept — it directly derives our standard loss functions. MSE comes from assuming Gaussian errors. Cross-entropy comes from assuming categorical labels. Adding a prior to MLE gives MAP estimation, which is equivalent to regularization: a Gaussian prior gives L2, a Laplace prior gives L1. Understanding this connects loss functions, distributions, and regularization into a single coherent framework."

---

## 4. Expectation, Variance, and Standard Deviation

### Expectation (Mean)

The average value you'd expect from a random variable over many samples:

```
E[X] = sum(x * P(X = x)) for all x    (discrete)
E[X] = integral(x * p(x) dx)           (continuous)
```

**ML context**:
- The *expected loss* over the data distribution is what we're really trying to minimize (we approximate it with the average loss over a mini-batch)
- Batch normalization computes running means (expectations) of activations
- The "bias" in bias-variance tradeoff is the expectation of your model's predictions minus the true value

### Variance

How spread out the values are around the mean:

```
Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
```

**ML context**:
- High variance in training loss = noisy gradients (might need larger batch size)
- Variance of predictions = model instability. High variance = overfitting
- **Bias-variance tradeoff**: Simple models have high bias, low variance. Complex models have low bias, high variance. The sweet spot is where total error is minimized.

### Standard Deviation

```
std(X) = sqrt(Var(X))
```

Same units as the original data (variance is in squared units, which is less interpretable).

**ML context**:
- Feature normalization: subtract mean, divide by std → `(x - mean) / std`
- Weight initialization: `W ~ N(0, std=1/sqrt(n))` — the std is chosen to keep activations from exploding or vanishing
- Reporting results: "accuracy of 94.2% +/- 1.3%" — the +/- is std over multiple runs

### The Bias-Variance Tradeoff

The expected prediction error decomposes into:

```
Error = Bias^2 + Variance + Irreducible Noise
```

| Model complexity | Bias | Variance | Total Error |
|---|---|---|---|
| Too simple (underfitting) | High | Low | High |
| Just right | Medium | Medium | Low |
| Too complex (overfitting) | Low | High | High |

A linear model fitting a quadratic relationship has high bias — it systematically gets the wrong answer. A degree-50 polynomial fitting 10 data points has high variance — it fits training data perfectly but is wildly different on new data.

**Modern deep learning nuance**: Very large neural networks seem to violate the classical tradeoff (the "double descent" phenomenon). They have low bias AND can achieve low variance with proper regularization. This is an active area of research and may come up in advanced interviews.

### Interview-Ready Explanation

> "Expectation is the average outcome, variance measures spread. In ML, the bias-variance tradeoff tells us that underfitting (high bias) and overfitting (high variance) are two sides of the same coin. Regularization, dropout, and early stopping reduce variance at the cost of slightly increased bias. Modern large models challenge the classical tradeoff but the intuition remains useful for understanding model behavior."

---

### Check Your Understanding

1. You are using MLE to train a linear regression model. If you assume the errors follow a Gaussian distribution, you get MSE loss. If you instead assume the errors follow a Laplace distribution, what loss function do you get?
2. Your model has high training accuracy but low test accuracy. In terms of the bias-variance decomposition, what is the dominant error term? Name two techniques to reduce it.
3. A colleague says "L2 regularization is just a hack to prevent overfitting — it has no theoretical basis." How would you respond using the MAP estimation framework?

<details>
<summary>Answers</summary>

1. A Laplace distribution has the density `p(x) = (1/2b) * exp(-|x - mu| / b)`. Taking the negative log-likelihood gives a term proportional to `|y_pred - y_true|`, which is the Mean Absolute Error (MAE / L1 loss). So Laplace errors lead to MAE, just as Gaussian errors lead to MSE.
2. The dominant error term is variance — the model is overfitting to training data and not generalizing. Techniques to reduce variance: (a) regularization (L2/dropout), (b) early stopping, (c) more training data, (d) reducing model complexity, (e) data augmentation.
3. L2 regularization has a rigorous Bayesian interpretation: it corresponds to MAP estimation with a Gaussian prior on the weights (`P(w) ~ N(0, sigma^2)`). The regularization term `lambda * ||w||^2` is exactly the negative log of this Gaussian prior. It encodes the belief that weights should be small, which is a principled inductive bias — not a hack.

</details>

---

## 5. Hypothesis Testing and A/B Testing ML Models

### The Basic Framework

You have two models (or a model vs. baseline). Are the performance differences real or just noise?

1. **Null hypothesis (H0)**: There's no real difference. The observed difference is due to random chance.
2. **Alternative hypothesis (H1)**: There IS a real difference.
3. **Compute a test statistic**: Measures how far the observed result is from what H0 predicts.
4. **p-value**: Probability of seeing a result this extreme *if H0 were true*.
5. **Decision**: If p-value < significance level (typically 0.05), reject H0.

### A/B Testing ML Models in Practice

**Scenario**: You trained Model B and it gets 94.2% accuracy vs. Model A's 93.8%. Is that a real improvement or just noise?

**Approach 1: Paired t-test on k-fold cross-validation**
```python
from scipy import stats

# Run 10-fold CV for both models, get accuracy for each fold
model_a_scores = [0.935, 0.941, 0.929, ...]  # 10 values
model_b_scores = [0.942, 0.948, 0.937, ...]  # 10 values

t_stat, p_value = stats.ttest_rel(model_b_scores, model_a_scores)
if p_value < 0.05:
    print("Model B is significantly better")
```

**Approach 2: Bootstrap confidence intervals**
```python
# Resample predictions with replacement, compute metric many times
differences = []
for _ in range(10000):
    idx = np.random.choice(len(y_test), size=len(y_test), replace=True)
    acc_a = accuracy(y_test[idx], preds_a[idx])
    acc_b = accuracy(y_test[idx], preds_b[idx])
    differences.append(acc_b - acc_a)

ci_lower, ci_upper = np.percentile(differences, [2.5, 97.5])
# If 0 is not in the CI, the difference is significant
```

### Common Pitfalls

- **Multiple comparisons**: Testing 20 model variants? By chance, one will appear significant at p=0.05. Use Bonferroni correction or similar.
- **Data leakage**: Your test set must be completely separate from all training and tuning.
- **Metric choice**: Statistical significance in accuracy doesn't mean practical significance. A 0.1% accuracy improvement might not matter.
- **Distribution assumptions**: t-tests assume normality. With enough samples (bootstrap), this is less of a concern.

### Interview-Ready Explanation

> "When comparing models, observed performance differences might be due to chance. Statistical testing (paired t-test on cross-validation folds, or bootstrap confidence intervals) tells you whether the difference is significant. The p-value is the probability of seeing such a difference if the models were actually equal. In practice, I prefer bootstrap methods because they make fewer distributional assumptions and directly give you confidence intervals."

---

## 6. Information Theory

### Entropy: Measuring Surprise

Entropy measures the average "surprise" or "uncertainty" in a distribution. If you know exactly what's going to happen, entropy is zero. If anything could happen with equal probability, entropy is maximized.

```
H(X) = -sum(P(xi) * log(P(xi)))
```

**Intuition**: A fair coin has entropy `log(2) = 1 bit`. You need 1 bit to encode the outcome. A biased coin (90% heads) has lower entropy — outcomes are more predictable, so you need fewer bits on average.

**ML context**:
- A well-trained classifier should have *low* entropy when it's confident (peaked distribution, one class has high probability)
- A poorly trained classifier has *high* entropy (uniform-ish distribution, unsure about all classes)
- Decision trees use entropy (or Gini impurity) to decide where to split: pick the feature that reduces entropy the most

### Cross-Entropy: The Loss Function

Cross-entropy measures the "distance" between two distributions — the true distribution `p` and the predicted distribution `q`:

```
H(p, q) = -sum(p(xi) * log(q(xi)))
```

For classification where `p` is one-hot (the true label is class 3):

```
p = [0, 0, 1, 0, 0]              # true distribution
q = [0.05, 0.1, 0.7, 0.1, 0.05]  # model's prediction

H(p, q) = -(0*log(0.05) + 0*log(0.1) + 1*log(0.7) + 0*log(0.1) + 0*log(0.05))
         = -log(0.7)
         ≈ 0.357
```

Only the true class term survives. Cross-entropy loss = `-log(predicted probability of the correct class)`. If the model is confident and right (`q = 0.99`), loss is low (`-log(0.99) = 0.01`). If the model is confident and wrong (`q = 0.01`), loss is high (`-log(0.01) = 4.6`).

This is why cross-entropy punishes confident wrong predictions severely. A model that says "99% cat" when it's a dog gets hammered.

### KL Divergence: How Different Are Two Distributions?

```
KL(p || q) = sum(p(xi) * log(p(xi) / q(xi)))
           = H(p, q) - H(p)
```

KL divergence measures how much information is "lost" when you approximate the true distribution `p` with `q`.

**Key properties:**
- Always non-negative: `KL(p || q) >= 0`
- Zero only when `p = q`
- **Not symmetric**: `KL(p || q) != KL(q || p)` — it's not a true "distance"

**ML context:**
- **VAEs (Variational Autoencoders)**: The loss includes a KL divergence term that forces the latent distribution to be close to a standard normal
- **Knowledge distillation**: Minimize KL divergence between teacher and student model outputs
- **Policy gradient methods**: KL divergence constrains how much the policy can change per update (PPO uses a form of this)
- **Minimizing cross-entropy = minimizing KL divergence** (since `H(p)` is constant with respect to model parameters)

### Why Cross-Entropy and Not MSE for Classification?

Two reasons:

1. **Gradient strength**: MSE with sigmoid gives gradients proportional to `sigmoid'(x)`, which is tiny when the model is confidently wrong. Cross-entropy's gradient is proportional to `(predicted - true)`, which is large when the model is wrong. Cross-entropy learns faster from mistakes.

2. **Probabilistic correctness**: Cross-entropy is the correct loss under the categorical likelihood assumption (MLE). MSE assumes Gaussian errors, which doesn't make sense for class probabilities.

```
Model predicts 0.001 for the true class:

MSE gradient:     small (sigmoid derivative is tiny in saturated region)
CE gradient:      large (-1/0.001 = -1000, strong signal to fix this)
```

### Interview-Ready Explanation

> "Entropy measures uncertainty in a distribution. Cross-entropy measures the gap between predicted and true distributions — it's the standard loss for classification because it penalizes confident wrong predictions heavily and has strong gradients even when the model is far from correct. KL divergence measures how different two distributions are and appears in VAEs (forcing latent distributions toward a prior), knowledge distillation, and policy optimization. Minimizing cross-entropy is equivalent to minimizing KL divergence from the true distribution."

---

### Check Your Understanding

1. A model predicts `P(correct class) = 0.01` (confidently wrong). What is the cross-entropy loss for this prediction? Compare it to the loss when `P(correct class) = 0.99`. Why does this asymmetry help training?
2. KL divergence is not symmetric: `KL(p || q) != KL(q || p)`. In knowledge distillation, we minimize `KL(teacher || student)`. What would happen differently if we minimized `KL(student || teacher)` instead?
3. A colleague argues that MSE should work fine for a 10-class classification problem. Using the concept of gradient strength, explain why cross-entropy is much better in practice.

<details>
<summary>Answers</summary>

1. When `P = 0.01`: loss = `-log(0.01) = 4.6`. When `P = 0.99`: loss = `-log(0.99) = 0.01`. The ratio is roughly 460:1. This steep penalty for confident wrong predictions creates strong gradients that force the model to rapidly correct its worst mistakes, prioritizing them over refining already-correct predictions.
2. `KL(teacher || student)` is "mode-covering" — the student is penalized whenever the teacher assigns high probability to something the student does not. This forces the student to cover all modes of the teacher's distribution. `KL(student || teacher)` is "mode-seeking" — the student can ignore modes of the teacher and concentrate mass on the teacher's peaks. For distillation, mode-covering is preferred because we want the student to learn the full distribution of the teacher, not just its most confident predictions.
3. With MSE on sigmoid/softmax outputs, when the model is confidently wrong (output near 0 for the true class), the sigmoid is in its saturated region where the derivative is nearly zero. The gradient `2 * (predicted - true) * sigmoid'(logit)` is tiny because `sigmoid'` is tiny. Cross-entropy's gradient is proportional to `(predicted - true)` without the sigmoid derivative factor, so it remains large when the model is wrong. This means cross-entropy provides strong learning signals exactly when they are most needed.

</details>

---

## 7. Important Distributions in ML Practice

### When to Use What

| Distribution | Use Case | Example |
|---|---|---|
| Bernoulli | Binary outcomes | Spam/not spam, click/no click |
| Categorical | Multi-class | Image classification (10 classes) |
| Normal (Gaussian) | Continuous values, errors | Regression targets, weight init |
| Poisson | Count data | Number of events per time period |
| Uniform | Equally likely outcomes | Random initialization, random sampling |
| Beta | Probabilities of probabilities | Prior in Bayesian binary models |
| Dirichlet | Distribution over distributions | Topic models (LDA) |

### The Reparameterization Trick

Used in VAEs. Problem: you can't backpropagate through a random sampling operation. Solution: instead of sampling `z ~ N(mu, sigma^2)`, reparameterize as:

```
epsilon ~ N(0, 1)
z = mu + sigma * epsilon
```

Now `z` is a deterministic function of `mu` and `sigma` (with `epsilon` as external randomness), so you CAN compute gradients with respect to `mu` and `sigma`. This trick enables training of VAEs and many other models with stochastic components.

### Interview-Ready Explanation

> "The choice of output distribution determines both your activation function and loss. Binary → sigmoid + binary cross-entropy. Multi-class → softmax + categorical cross-entropy. Continuous → linear + MSE (Gaussian assumption). These aren't arbitrary pairings — each combination comes from MLE under the corresponding distributional assumption."

---

## 8. Practical Statistical Thinking for ML

### Calibration

A model that says "80% probability" should be right 80% of the time. Many models aren't calibrated — they're overconfident or underconfident.

- **Temperature scaling**: A simple post-hoc calibration method. Scale logits by a learned temperature before softmax.
- **Platt scaling**: Fit a logistic regression on top of model scores.
- **Reliability diagrams**: Plot predicted probability vs. actual frequency. A perfectly calibrated model follows the diagonal.

### Confidence Intervals for Model Performance

Never report a single accuracy number without uncertainty:

```python
# Bootstrap 95% CI for accuracy:
accuracies = []
for _ in range(10000):
    idx = np.random.choice(len(y_test), size=len(y_test), replace=True)
    accuracies.append(accuracy_score(y_test[idx], preds[idx]))
lower, upper = np.percentile(accuracies, [2.5, 97.5])
print(f"Accuracy: {np.mean(accuracies):.3f} ({lower:.3f} - {upper:.3f})")
```

### Class Imbalance: A Statistical Problem

When 99% of your data is one class, accuracy is misleading (predict majority class = 99% accuracy, useless model). You need:

- **Precision**: Of things predicted positive, how many actually are? `TP / (TP + FP)`
- **Recall**: Of actual positives, how many did we find? `TP / (TP + FN)`
- **F1 score**: Harmonic mean of precision and recall
- **AUROC**: Area under the ROC curve — threshold-independent measure
- **PR-AUC**: Area under the precision-recall curve — better for severe imbalance

### Interview-Ready Explanation

> "Statistical thinking in ML means never trusting a single number. Report confidence intervals. Check calibration. Use appropriate metrics for imbalanced data. Understand that cross-validation gives you a distribution of scores, not a single score, and make decisions based on statistical significance — not just which number looks bigger."

---

## Key Takeaways

1. **Probability gives ML its language.** Model outputs are distributions. Loss functions come from likelihoods. Regularization comes from priors.
2. **MLE is the bridge.** It connects distributions to loss functions: Gaussian → MSE, Categorical → cross-entropy.
3. **Bayes' theorem flips probabilities.** We want `P(class | data)` but can estimate `P(data | class)`.
4. **Cross-entropy is THE classification loss** because it has strong gradients and is probabilistically correct.
5. **KL divergence measures distribution gaps.** Used in VAEs, distillation, and RL.
6. **Statistical testing prevents false conclusions.** Always test whether observed improvements are significant.
7. **Calibration matters.** A model's probabilities should reflect actual frequencies.

Every design choice in ML has a probabilistic interpretation. Understanding that interpretation makes you a fundamentally better ML engineer.

---

## Common Pitfalls

**Pitfall 1: Using Accuracy as the Sole Metric for Imbalanced Data**
- Symptom: Model reports 99% accuracy on a fraud detection task but catches zero actual fraud
- Why: When the positive class is 1% of the data, always predicting "not fraud" gives 99% accuracy. Accuracy rewards predicting the majority class and hides the model's failure on the minority class.
- Fix: Use precision, recall, F1, AUROC, or PR-AUC. For severe imbalance, PR-AUC is especially informative because it focuses on performance on the positive class.

**Pitfall 2: Treating Softmax Probabilities as Calibrated Confidences**
- Symptom: A model says "95% confident" but is actually correct only 70% of the time for such predictions
- Why: Neural networks are often overconfident — softmax outputs are not inherently calibrated. The model maximizes cross-entropy, which pushes probabilities toward 0 and 1 but does not guarantee calibration.
- Fix: Apply post-hoc calibration (temperature scaling or Platt scaling). Evaluate calibration with reliability diagrams. Never trust raw softmax probabilities as true confidence estimates.

**Pitfall 3: Claiming Model Improvements Without Statistical Testing**
- Symptom: "Model B gets 94.2% vs. Model A's 93.8% — Model B is better!"
- Why: A 0.4% difference could easily be due to random variation in the test set or data split. Without statistical testing, you cannot distinguish signal from noise.
- Fix: Use paired t-tests on cross-validation folds or bootstrap confidence intervals. If the confidence interval for the difference includes zero, the improvement is not statistically significant.

**Pitfall 4: Multiple Comparisons Without Correction**
- Symptom: You test 20 model variants against a baseline, and one achieves p < 0.05. You declare it significantly better.
- Why: With 20 independent tests at significance level 0.05, you expect 1 false positive purely by chance (`20 * 0.05 = 1`). This is the multiple comparisons problem.
- Fix: Apply Bonferroni correction (divide significance level by number of tests: `0.05 / 20 = 0.0025`) or use False Discovery Rate (FDR) control.

---

## Hands-On Exercises

### Exercise 1: MLE Connects Distributions to Loss Functions
**Goal:** Verify by hand and code that MSE and cross-entropy emerge from MLE under different distributional assumptions.
**Task:**
1. Generate 100 data points from a linear model with Gaussian noise: `y = 3*x + 2 + N(0, 0.5)`.
2. Write the Gaussian log-likelihood function and show algebraically that maximizing it is equivalent to minimizing MSE.
3. Implement both the log-likelihood maximization and MSE minimization in NumPy. Verify they give the same optimal parameters.
4. Now generate binary labels from a logistic model: `P(y=1 | x) = sigmoid(2*x - 1)`. Write the Bernoulli log-likelihood and show it equals (negative) binary cross-entropy.
**Verify:** The optimal parameters from log-likelihood maximization should match those from loss minimization to within numerical precision.

### Exercise 2: Bootstrap Confidence Intervals for Model Comparison
**Goal:** Learn to statistically compare two models rather than relying on single-number comparisons.
**Task:**
1. Train two sklearn classifiers (e.g., logistic regression and random forest) on a dataset of your choice.
2. Generate predictions on a held-out test set.
3. Implement a bootstrap procedure: resample the test set 10,000 times with replacement, computing the accuracy difference between the two models each time.
4. Compute the 95% confidence interval for the accuracy difference.
5. Determine whether the difference is statistically significant (does the CI include zero?).
**Verify:** Your bootstrap CI should be consistent with a paired t-test on cross-validation folds (both methods should agree on significance).

---

## 9. Sampling and Monte Carlo Methods

### Why Sampling Matters

Many quantities in ML are expectations over distributions we can't compute analytically. The solution: draw samples and average.

```
E[f(x)] ≈ (1/N) * sum(f(x_i))    where x_i ~ P(x)
```

This is the **Monte Carlo** principle: approximate an intractable integral by averaging over random samples.

### Where Sampling Appears in ML

**Training itself**: SGD doesn't compute the true gradient (over ALL data). It samples a mini-batch and computes an approximate gradient. This is a Monte Carlo estimate of the true gradient.

**Dropout**: At train time, randomly zero out neurons. This is sampling from an ensemble of sub-networks. At test time, you approximate the ensemble average.

**Monte Carlo Dropout**: Run inference multiple times WITH dropout enabled. The variance of predictions estimates model uncertainty. Cheap Bayesian inference.

```python
# Monte Carlo Dropout for uncertainty estimation
model.train()  # keep dropout active
predictions = [model(x) for _ in range(100)]
mean_pred = torch.stack(predictions).mean(dim=0)
uncertainty = torch.stack(predictions).std(dim=0)
```

**MCMC (Markov Chain Monte Carlo)**: When you can't sample directly from a distribution, construct a Markov chain whose stationary distribution IS the target. Used in Bayesian deep learning and probabilistic programming (PyMC, Stan).

**Importance Sampling**: When sampling from P is hard, sample from an easier distribution Q and reweight. Used in off-policy reinforcement learning and rare event estimation.

### The Law of Large Numbers

More samples = better approximation. The error shrinks as `1/sqrt(N)`. To halve the error, you need 4x the samples. This is why larger batch sizes give more stable gradients but with diminishing returns.

### Interview Question

> "Your model gives a single prediction for each input. Your product manager wants confidence estimates. You can't retrain the model. What do you do?"
>
> Monte Carlo Dropout: run the existing model multiple times with dropout enabled at inference. The spread of predictions gives you uncertainty. Alternatively, if the model outputs logits, calibrate them with temperature scaling or Platt scaling so the softmax probabilities reflect true confidence.

---

## Test Yourself

1. **A spam classifier outputs P(spam) = 0.7. Using Bayes' theorem, explain what prior information could shift this to 0.3 after considering that the sender is in the user's contacts.**

2. **You're choosing a loss function for a regression task where occasional large errors are acceptable but consistent small errors are not. MSE or MAE? Why, using distributional reasoning?**

3. **Your model has 99.1% accuracy on a fraud detection task. The positive rate is 0.5%. Is this model useful? What metrics should you look at instead?**

4. **Explain why minimizing cross-entropy is equivalent to minimizing KL divergence from the true distribution. What's the practical implication?**

5. **In a VAE, the loss has a reconstruction term and a KL divergence term. What happens if you weight the KL term too heavily? Too lightly?**

6. **You run an A/B test comparing two models. Model B wins with p=0.03. You then test Model C vs Model A and get p=0.04. Can you claim both B and C are significantly better? Why or why not?**

7. **Using MLE reasoning, derive why binary cross-entropy is the correct loss for a binary classifier with sigmoid output.**

8. **You want to estimate the expected revenue of a new recommendation model before deploying it. You have logged data from the current model. Explain how importance sampling could help.**

---

## Summary

Every design choice in ML has a probabilistic interpretation: loss functions come from likelihood assumptions, regularization comes from priors, and model outputs are distributions. The key unifying insight is MLE — it bridges the gap between probability distributions and the loss functions you minimize, turning "why cross-entropy?" from an arbitrary convention into a principled derivation. Statistical thinking (calibration, confidence intervals, hypothesis testing) is what separates reliable ML engineering from guessing.

## What's Next

- **Next lesson:** [Optimization for Machine Learning](../optimization/COURSE.md) — covers gradient descent, learning rates, and regularization techniques, directly applying the loss functions and probabilistic principles from this lesson
- **Builds on this:** [Classical ML: Supervised Learning](../../04-classical-ml/supervised/COURSE.md) — uses distributions and MLE to understand logistic regression, Naive Bayes, and model evaluation
- **Deep dive:** [LLM Internals: Pretraining](../../03-llm-internals/pretraining/COURSE.md) — applies cross-entropy loss, sampling, and distributional thinking to language model training at scale
