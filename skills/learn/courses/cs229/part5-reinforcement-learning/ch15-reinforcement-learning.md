# Chapter 15: Reinforcement Learning

## Introduction

RL learns optimal behavior through trial-and-error interaction with an environment, receiving only reward signals (no labeled examples).

**Examples**:
- Game playing (chess, Go, Atari)
- Robotics (walking, manipulation)
- Resource management
- Recommendation systems

## 15.1 Markov Decision Processes

### Definition

An MDP is a tuple (S, A, {P_{sa}}, γ, R):
- **S**: Set of states
- **A**: Set of actions
- **P_{sa}**: Transition probability P(s'|s, a)
- **γ ∈ [0, 1)**: Discount factor
- **R: S → ℝ**: Reward function (can also depend on a)

### The Markov Property

Future depends only on current state, not history:
```
P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ...) = P(s_{t+1} | s_t, a_t)
```

### Policy

A policy π: S → A maps states to actions.

**Deterministic**: π(s) = a

**Stochastic**: π(a|s) = P(a|s)

### Value Function

Expected cumulative discounted reward from state s:
```
V^π(s) = E[R(s_0) + γR(s_1) + γ²R(s_2) + ... | s_0 = s, π]
```

### Bellman Equation

For policy π:
```
V^π(s) = R(s) + γ Σ_{s'} P_{sπ(s)}(s') V^π(s')
```

**Intuition**: Value = immediate reward + discounted future value.

### Optimal Value Function

```
V*(s) = max_π V^π(s)
```

**Bellman optimality equation**:
```
V*(s) = R(s) + γ max_a Σ_{s'} P_{sa}(s') V*(s')
```

### Optimal Policy

Given V*, the optimal policy is:
```
π*(s) = argmax_a Σ_{s'} P_{sa}(s') V*(s')
```

## 15.2 Value Iteration and Policy Iteration

### Value Iteration

Iteratively apply Bellman optimality:

```
Initialize V(s) = 0 for all s

Repeat until convergence:
    For each state s:
        V(s) := R(s) + γ max_a Σ_{s'} P_{sa}(s') V(s')
```

**Convergence**: V converges to V* (contraction mapping).

**Synchronous vs Asynchronous**:
- Synchronous: Update all states at once
- Asynchronous: Update one state at a time (often faster)

### Policy Iteration

Alternate between evaluation and improvement:

```
Initialize π randomly

Repeat until convergence:
    (a) Policy Evaluation: Compute V^π
        Solve: V^π(s) = R(s) + γ Σ_{s'} P_{sπ(s)}(s') V^π(s')
        (Linear system of |S| equations)
    
    (b) Policy Improvement: Update policy
        For each s: π(s) := argmax_a Σ_{s'} P_{sa}(s') V^π(s')
```

**Convergence**: Finite steps for finite MDPs.

### Comparison

| Aspect | Value Iteration | Policy Iteration |
|--------|-----------------|------------------|
| **Per-iteration cost** | O(\|S\|²\|A\|) | O(\|S\|³ + \|S\|²\|A\|) |
| **Convergence** | Asymptotic | Finite steps |
| **Best for** | Large state spaces | Small/medium MDPs |

## 15.3 Learning a Model for an MDP

When P_{sa} and R are unknown, learn them from experience.

### Estimating Transition Probabilities

From observed transitions:
```
P̂_{sa}(s') = (# times action a in state s led to s') / (# times action a taken in state s)
```

If no data: P̂_{sa}(s') = 1/|S| (uniform).

### Model-Based RL Algorithm

```
1. Initialize π randomly

2. Repeat:
   a. Execute π in MDP, collect experience
   b. Update P̂_{sa} (and R̂ if needed) from experience
   c. Apply value/policy iteration with estimated model
   d. Update π to be greedy with respect to new V
```

**Warm start**: Initialize value iteration with previous solution.

## 15.4 Continuous State MDPs

State space S = ℝᵈ (e.g., robot position and velocity).

### 15.4.1 Discretization

Divide state space into grid cells, treat each as discrete state.

```
[   ][   ][   ]
[   ][   ][   ]  → Discrete MDP with 9 states
[   ][   ][   ]
```

**Problems**:
1. Piecewise constant value function (doesn't generalize)
2. Curse of dimensionality: k^d cells for d dimensions

**Rule of thumb**: Works for ≤ 4 dimensions with care.

### 15.4.2 Value Function Approximation

Approximate V(s) with a parameterized function V_θ(s).

**Common choices**:
- Linear: V_θ(s) = θᵀφ(s) (features φ)
- Neural network: V_θ(s) = NN(s; θ)

### Fitted Value Iteration

```
1. Sample states: s^(1), ..., s^(m)

2. Repeat:
   For each sampled state s^(i):
       Compute target: y^(i) = R(s^(i)) + γ max_a E_{s'~P}[V(s')]
                             ≈ R(s^(i)) + γ max_a V(simulator(s^(i), a))
   
   Fit V_θ to targets using supervised learning:
       θ := argmin_θ Σ_i (V_θ(s^(i)) - y^(i))²
```

**Key**: Use simulator/model to compute next-state expectations.

### Challenges

1. **Function approximation error**: V_θ may not represent V* well
2. **Distribution shift**: Sampled states may not cover important regions
3. **Deadly triad**: Off-policy + function approximation + bootstrapping → instability

## 15.5 Connections Between Value and Policy Iteration

### Policy Iteration as Modified Value Iteration

Policy iteration with 1 step of evaluation = value iteration.

Policy iteration with ∞ steps of evaluation = standard policy iteration.

**Generalized Policy Iteration**: Any mixture works!

### The Actor-Critic Framework

- **Critic**: Estimates value function V^π
- **Actor**: Improves policy using critic's estimates

This generalizes both value and policy iteration.

## Key Takeaways

1. **MDPs** formalize sequential decision-making
2. **Bellman equations** decompose value recursively
3. **Value iteration**: Iterates Bellman optimality
4. **Policy iteration**: Alternates evaluation and improvement
5. **Model-based**: Learn P_{sa}, then plan
6. **Function approximation**: Handles continuous states

## Practical Notes

- **Discount factor γ**: Higher = more farsighted (0.99 common)
- **Exploration**: ε-greedy, softmax, or UCB
- **Deep RL**: DQN, PPO, SAC for complex environments
- **Simulation**: Often essential for sample efficiency
- **Reward shaping**: Design rewards carefully!

