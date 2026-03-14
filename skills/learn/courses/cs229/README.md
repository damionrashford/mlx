# CS229: Machine Learning - Stanford University

**Authors:** Andrew Ng and Tengyu Ma  
**Course Date:** Spring 2023 (Updated June 11, 2023)

## Course Overview

This is the complete lecture notes for Stanford's CS229 Machine Learning course - one of the most comprehensive and rigorous introductions to machine learning theory and practice. The course provides the mathematical foundations underlying modern ML and AI systems.

## Course Structure

The course is organized into **5 major parts** containing **17 chapters**:

### Part I: Supervised Learning (Chapters 1-6)
Foundation of predictive modeling - learn to map inputs to outputs.

| Chapter | Topic | Key Concepts |
|---------|-------|--------------|
| 1 | Linear Regression | LMS, Normal Equations, MLE Interpretation |
| 2 | Classification & Logistic Regression | Sigmoid, Cross-Entropy, Perceptron |
| 3 | Generalized Linear Models | Exponential Family, GLM Design |
| 4 | Generative Learning Algorithms | GDA, Naive Bayes, Discriminative vs Generative |
| 5 | Kernel Methods | Feature Maps, Kernel Trick, Mercer's Theorem |
| 6 | Support Vector Machines | Margins, Lagrange Duality, SMO Algorithm |

### Part II: Deep Learning (Chapter 7)
Neural networks and modern deep learning architectures.

| Chapter | Topic | Key Concepts |
|---------|-------|--------------|
| 7 | Deep Learning | MLPs, Activation Functions, Backpropagation, Conv/Norm Layers |

### Part III: Generalization & Regularization (Chapters 8-9)
Understanding why models work and how to make them work better.

| Chapter | Topic | Key Concepts |
|---------|-------|--------------|
| 8 | Generalization | Bias-Variance Tradeoff, Double Descent, VC Dimension |
| 9 | Regularization & Model Selection | L1/L2 Regularization, Cross-Validation, Bayesian View |

### Part IV: Unsupervised Learning (Chapters 10-14)
Discovering structure in unlabeled data.

| Chapter | Topic | Key Concepts |
|---------|-------|--------------|
| 10 | Clustering | K-Means Algorithm |
| 11 | EM Algorithms | Mixture Models, Jensen's Inequality, ELBO |
| 12 | Principal Components Analysis | Dimensionality Reduction, Eigenvectors |
| 13 | Independent Components Analysis | Blind Source Separation |
| 14 | Self-Supervised Learning & Foundation Models | Pretraining, Transformers, LLMs |

### Part V: Reinforcement Learning & Control (Chapters 15-17)
Learning to make decisions through interaction.

| Chapter | Topic | Key Concepts |
|---------|-------|--------------|
| 15 | Reinforcement Learning | MDPs, Value/Policy Iteration, Q-Learning |
| 16 | LQR, DDP, LQG | Optimal Control, Kalman Filtering |
| 17 | Policy Gradient (REINFORCE) | Model-Free RL, Gradient Estimation |

## Directory Structure

```
cs229-stanford-ml/
├── README.md                            # This overview file
│
├── part1-supervised-learning/           # Chapters 1-6
│   ├── INDEX.md
│   ├── ch01-linear-regression.md
│   ├── ch02-logistic-regression.md
│   ├── ch03-generalized-linear-models.md
│   ├── ch04-generative-algorithms.md
│   ├── ch05-kernel-methods.md
│   └── ch06-support-vector-machines.md
│
├── part2-deep-learning/                 # Chapter 7
│   ├── INDEX.md
│   └── ch07-deep-learning.md
│
├── part3-generalization/                # Chapters 8-9
│   ├── INDEX.md
│   ├── ch08-generalization.md
│   └── ch09-regularization.md
│
├── part4-unsupervised-learning/         # Chapters 10-14
│   ├── INDEX.md
│   ├── ch10-clustering.md
│   ├── ch11-em-algorithms.md
│   ├── ch12-pca.md
│   ├── ch13-ica.md
│   └── ch14-foundation-models.md
│
└── part5-reinforcement-learning/        # Chapters 15-17
    ├── INDEX.md
    ├── ch15-reinforcement-learning.md
    ├── ch16-lqr-ddp-lqg.md
    └── ch17-policy-gradient.md
```

**Note**: If you see `COURSE_INDEX.md`, `part1-supervised/`, or `part4-unsupervised/` folders, 
those are old stubs that can be deleted.

## Prerequisites

- **Linear Algebra**: Matrices, eigenvectors, matrix calculus
- **Probability & Statistics**: Distributions, MLE, Bayes' theorem
- **Multivariable Calculus**: Gradients, Hessians, chain rule
- **Programming**: Python (NumPy, scikit-learn)

## Learning Path for Applied ML Engineers

### Priority 1: Core Foundations
1. Chapter 1: Linear Regression (optimization basics)
2. Chapter 2: Logistic Regression (classification)
3. Chapter 8: Generalization (bias-variance, why models fail)
4. Chapter 9: Regularization (preventing overfitting)

### Priority 2: Production ML
5. Chapter 6: SVMs (maximum margin, kernel trick)
6. Chapter 7: Deep Learning (neural networks)
7. Chapter 10-12: Unsupervised (clustering, dimensionality reduction)

### Priority 3: Advanced Topics
8. Chapter 14: Foundation Models (modern LLMs, Transformers)
9. Chapter 15-17: Reinforcement Learning (decision making)

## Key Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| x^(i) | i-th training example (input) |
| y^(i) | i-th training example (output/label) |
| θ | Model parameters |
| h_θ(x) | Hypothesis function |
| J(θ) | Cost/Loss function |
| ∇_θ | Gradient with respect to θ |
| X | Design matrix (n × d) |
| n | Number of training examples |
| d | Number of features |

## Source

Original PDF: CS229 Lecture Notes by Andrew Ng and Tengyu Ma, Stanford University

