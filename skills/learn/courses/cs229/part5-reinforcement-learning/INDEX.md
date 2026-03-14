# Part V: Reinforcement Learning and Control

## Overview

**Reinforcement Learning (RL)** learns to make sequential decisions through interaction with an environment. Unlike supervised learning, there are no labeled examples - only rewards/penalties based on actions taken.

## Chapters in This Part

- **[Chapter 15: Reinforcement Learning](ch15-reinforcement-learning.md)** - MDPs, value/policy iteration
- **[Chapter 16: LQR, DDP, and LQG](ch16-lqr-ddp-lqg.md)** - Optimal control theory
- **[Chapter 17: Policy Gradient (REINFORCE)](ch17-policy-gradient.md)** - Model-free RL

## The RL Problem

### Agent-Environment Interaction

```
     ┌─────────────┐
     │   Agent     │
     │  (policy π) │
     └─────┬───────┘
           │ action a
           ▼
     ┌─────────────┐
     │ Environment │──→ state s', reward r
     └─────────────┘
```

At each step:
1. Agent observes state s
2. Agent takes action a = π(s)
3. Environment transitions to s' and gives reward r
4. Repeat

**Goal**: Find policy π that maximizes cumulative reward.

## Core Concepts

### Markov Decision Process (MDP)

An MDP is defined by:
- **S**: State space
- **A**: Action space
- **P_{sa}**: Transition probabilities P(s'|s, a)
- **R**: Reward function R(s, a)
- **γ**: Discount factor (0 ≤ γ < 1)

### Value Functions

**State value** V^π(s): Expected return starting from s, following π
```
V^π(s) = E[Σ_{t=0}^∞ γ^t R(s_t, a_t) | s_0 = s, π]
```

**Action value** Q^π(s, a): Expected return starting from s, taking a, then following π
```
Q^π(s, a) = E[Σ_{t=0}^∞ γ^t R(s_t, a_t) | s_0 = s, a_0 = a, π]
```

### Bellman Equations

**Bellman expectation** (for fixed policy π):
```
V^π(s) = R(s, π(s)) + γ Σ_{s'} P_{sπ(s)}(s') V^π(s')
```

**Bellman optimality** (for optimal V*):
```
V*(s) = max_a [R(s, a) + γ Σ_{s'} P_{sa}(s') V*(s')]
```

## Key Algorithms

| Algorithm | Type | Updates |
|-----------|------|---------|
| **Value Iteration** | Model-based | Iterate Bellman optimality |
| **Policy Iteration** | Model-based | Evaluate → Improve → Repeat |
| **Q-Learning** | Model-free | TD update on Q |
| **SARSA** | Model-free | On-policy TD |
| **Policy Gradient** | Model-free | Gradient on policy parameters |
| **Actor-Critic** | Model-free | Combine value and policy |

## Control Theory Perspective

### Linear Quadratic Regulator (LQR)

For linear dynamics and quadratic costs:
```
s_{t+1} = As_t + Ba_t
cost = Σ_t (s_t^T U s_t + a_t^T W a_t)
```

**Optimal policy is linear**: a* = -Ls

### When to Use What

| Setting | Method |
|---------|--------|
| **Known model, linear** | LQR |
| **Known model, nonlinear** | DDP, MPC |
| **Unknown model, discrete** | Q-learning |
| **Unknown model, continuous** | Policy gradient, SAC |
| **Partial observability** | POMDP methods, LQG |

## Why RL is Hard

1. **Credit assignment**: Which actions led to reward?
2. **Exploration vs exploitation**: Try new things or use what works?
3. **Sample efficiency**: Learning from limited interaction
4. **Stability**: Value function approximation can diverge
5. **Partial observability**: Real world has hidden state
