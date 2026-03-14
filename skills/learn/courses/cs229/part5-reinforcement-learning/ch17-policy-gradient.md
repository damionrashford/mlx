# Chapter 17: Policy Gradient (REINFORCE)

## Introduction

**Policy gradient** methods directly optimize the policy without learning value functions. They are **model-free** - no need to know transition probabilities.

**Key advantage**: Works with continuous actions and stochastic policies.

## Setup

### Stochastic Policy

Policy π_θ(a|s) gives probability of action a in state s, parameterized by θ.

**Examples**:
- Discrete actions: Softmax over action logits
- Continuous actions: Gaussian with learned mean and variance

### Objective

Maximize expected total payoff:
```
η(θ) = E[Σ_{t=0}^{T-1} γ^t R(s_t, a_t)]
```

Where trajectories τ = (s_0, a_0, s_1, a_1, ..., s_T) are sampled by running π_θ.

### The Challenge

How to compute ∇_θ η(θ) when:
- We don't know the dynamics P_{sa}
- We can only sample trajectories

## Policy Gradient Derivation

### Key Identity

For any distribution P_θ dependent on θ:
```
∇_θ E_{τ~P_θ}[f(τ)] = E_{τ~P_θ}[f(τ) · ∇_θ log P_θ(τ)]
```

**Proof** (log-derivative trick):
```
∇_θ ∫ P_θ(τ) f(τ) dτ = ∫ ∇_θ P_θ(τ) · f(τ) dτ
                      = ∫ P_θ(τ) · (∇_θ P_θ(τ) / P_θ(τ)) · f(τ) dτ
                      = E_{τ~P_θ}[f(τ) · ∇_θ log P_θ(τ)]
```

### Trajectory Probability

```
P_θ(τ) = μ(s_0) · Π_{t=0}^{T-1} [π_θ(a_t|s_t) · P_{s_t a_t}(s_{t+1})]
```

Taking log:
```
log P_θ(τ) = log μ(s_0) + Σ_t [log π_θ(a_t|s_t) + log P_{s_t a_t}(s_{t+1})]
```

### The Gradient

Taking ∇_θ (dynamics terms disappear - they don't depend on θ!):
```
∇_θ log P_θ(τ) = Σ_{t=0}^{T-1} ∇_θ log π_θ(a_t|s_t)
```

### The Policy Gradient Theorem

```
∇_θ η(θ) = E_{τ~P_θ}[(Σ_{t=0}^{T-1} ∇_θ log π_θ(a_t|s_t)) · (Σ_{t=0}^{T-1} γ^t R(s_t, a_t))]
```

**Key insight**: We can estimate this using sampled trajectories!

## The REINFORCE Algorithm

### Monte Carlo Estimate

Sample n trajectories τ^(1), ..., τ^(n), estimate gradient:
```
∇_θ η(θ) ≈ (1/n) Σ_{i=1}^n [(Σ_t ∇_θ log π_θ(a_t^(i)|s_t^(i))) · f(τ^(i))]
```

Where f(τ) = Σ_t γ^t R(s_t, a_t) is the return.

### Algorithm

```
Initialize θ randomly

Repeat:
    1. Collect n trajectories by running π_θ
    
    2. For each trajectory τ^(i):
       - Compute return: f(τ^(i)) = Σ_t γ^t R(s_t^(i), a_t^(i))
       - Compute gradient: g^(i) = (Σ_t ∇_θ log π_θ(a_t^(i)|s_t^(i))) · f(τ^(i))
    
    3. Update: θ := θ + α · (1/n) Σ_i g^(i)
```

## Interpretation

### Why Does This Work?

The gradient (Σ_t ∇_θ log π_θ(a_t|s_t)) · f(τ) has intuition:

- **Σ_t ∇_θ log π_θ(a_t|s_t)**: Direction to make actions more likely
- **f(τ)**: Weight by trajectory quality

**Effect**: Increase probability of high-reward trajectories, decrease probability of low-reward ones.

### Baseline Subtraction

**Observation**: E[Σ_t ∇_θ log π_θ(a_t|s_t)] = 0

**Implication**: We can subtract any baseline b without changing expectation:
```
∇_θ η(θ) = E[(Σ_t ∇_θ log π_θ(a_t|s_t)) · (f(τ) - b)]
```

**Common choice**: b = E[f(τ)] (average return)

**Benefit**: Reduces variance of gradient estimates!

### Reward-to-Go

Further refinement: Action at time t only affects future rewards.

```
∇_θ η(θ) = E[Σ_t ∇_θ log π_θ(a_t|s_t) · (Σ_{t'≥t} γ^{t'-t} R(s_{t'}, a_{t'}))]
```

Use **reward-to-go** instead of full return → lower variance.

## Variance Reduction Techniques

| Technique | Description |
|-----------|-------------|
| **Baseline** | Subtract mean return |
| **Reward-to-go** | Only future rewards |
| **Advantage** | A(s,a) = Q(s,a) - V(s) |
| **GAE** | Generalized advantage estimation |

## Actor-Critic Methods

Combine policy gradient (actor) with learned value function (critic):

**Critic**: Learn V_φ(s) ≈ V^π(s) via TD learning

**Actor**: Use V_φ as baseline/advantage estimate

**Advantage Actor-Critic (A2C)**:
```
∇_θ J ≈ E[Σ_t ∇_θ log π_θ(a_t|s_t) · (R_t + γV_φ(s_{t+1}) - V_φ(s_t))]
```

The term R_t + γV_φ(s_{t+1}) - V_φ(s_t) is the **TD error** (one-step advantage estimate).

## Key Takeaways

1. **Policy gradient** directly optimizes expected return
2. **Log-derivative trick** enables gradient estimation from samples
3. **REINFORCE** is a simple Monte Carlo policy gradient
4. **Baselines** reduce variance without changing expectation
5. **Actor-critic** combines policy and value learning

## Practical Notes

- **High variance**: REINFORCE needs many samples
- **Learning rate**: Critical hyperparameter, often use adaptive (Adam)
- **Continuous actions**: Gaussian policy with learned mean and std
- **Entropy bonus**: Encourages exploration (π not too deterministic)
- **PPO (Proximal Policy Optimization)**: Most popular modern algorithm
- **SAC (Soft Actor-Critic)**: Good for continuous control
- **Libraries**: Stable-Baselines3, RLlib, CleanRL

