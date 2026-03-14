# Chapter 16: LQR, DDP, and LQG

## Introduction

Control theory provides powerful tools when dynamics are known (or can be learned). This chapter covers optimal control for:
- **LQR**: Linear dynamics, quadratic costs
- **DDP**: Nonlinear dynamics (via linearization)
- **LQG**: Partial observability

## 16.1 Finite-Horizon MDPs

### Setup

Finite time horizon T with:
- **Dynamics**: s_{t+1} = f(s_t, a_t) + noise
- **Costs**: c_t(s_t, a_t) (minimize instead of maximize reward)
- **Terminal cost**: c_T(s_T)

**Goal**: Find policy π₀, π₁, ..., π_{T-1} minimizing:
```
E[Σ_{t=0}^{T-1} c_t(s_t, a_t) + c_T(s_T)]
```

### Time-Varying Policies

Unlike infinite-horizon, optimal policy may depend on time t:
```
a_t = π_t(s_t)
```

### Dynamic Programming

**Backward recursion**:
```
V_T(s) = c_T(s)

V_t(s) = min_a [c_t(s, a) + E_{s'~P}[V_{t+1}(s')]]
```

Optimal action:
```
π_t(s) = argmin_a [c_t(s, a) + E_{s'~P}[V_{t+1}(s')]]
```

## 16.2 Linear Quadratic Regulation (LQR)

### The LQR Model

**Linear dynamics**:
```
s_{t+1} = A_t s_t + B_t a_t + w_t,  where w_t ~ N(0, Σ_t)
```

**Quadratic costs**:
```
c_t(s, a) = s^T U_t s + a^T W_t a
c_T(s) = s^T U_T s
```

Where U_t, W_t are positive semi-definite matrices.

### Key Result: Optimal Policy is Linear

```
a*_t = -L_t s_t
```

The optimal policy is **linear** in the state!

### Value Function is Quadratic

```
V_t(s) = s^T Φ_t s + Ψ_t
```

### Discrete Riccati Equations

Backward recursion for Φ_t and L_t:
```
Φ_t = A_t^T [Φ_{t+1} - Φ_{t+1} B_t (B_t^T Φ_{t+1} B_t + W_t)^{-1} B_t^T Φ_{t+1}] A_t + U_t

L_t = (B_t^T Φ_{t+1} B_t + W_t)^{-1} B_t^T Φ_{t+1} A_t
```

### Key Properties

1. **Φ_t depends only on dynamics and costs** (not noise!)
2. **L_t is independent of noise** → certainty equivalence
3. **Convergence guaranteed** if system is controllable

### LQR Algorithm

```
1. (If needed) Estimate A_t, B_t, Σ_t from data

2. Initialize: Φ_T = U_T, Ψ_T = 0

3. Backward pass (t = T-1 to 0):
   - Update Φ_t using Riccati equation
   - Compute L_t

4. Forward pass (execute policy):
   - a_t = -L_t s_t
```

## 16.3 From Nonlinear Dynamics to LQR

### 16.3.1 Linearization

For nonlinear dynamics s_{t+1} = F(s_t, a_t), linearize around operating point (s̄, ā):
```
s_{t+1} ≈ F(s̄, ā) + ∇_s F(s̄, ā)(s_t - s̄) + ∇_a F(s̄, ā)(a_t - ā)
```

This gives:
```
s_{t+1} ≈ A s_t + B a_t + constant
```

where A = ∇_s F, B = ∇_a F.

### 16.3.2 Differential Dynamic Programming (DDP)

For trajectory following with nonlinear dynamics:

**Step 1**: Generate nominal trajectory with naive controller:
```
s*_0, a*_0 → s*_1, a*_1 → ... → s*_T
```

**Step 2**: Linearize dynamics around each (s*_t, a*_t):
```
s_{t+1} ≈ A_t s_t + B_t a_t
```

**Step 3**: Quadratize costs (2nd-order Taylor):
```
c(s_t, a_t) ≈ s^T U_t s + a^T W_t a + linear terms
```

**Step 4**: Apply LQR to get improved policy π_t

**Step 5**: Execute new policy to get new trajectory, go to Step 2

**Repeat** until convergence.

### DDP vs LQR

| LQR | DDP |
|-----|-----|
| Linear dynamics | Nonlinear dynamics |
| Single solution | Iterative refinement |
| Global optimum | Local optimum |
| Closed-form | Numerical |

## 16.4 Linear Quadratic Gaussian (LQG)

### Partial Observability

Often we don't observe the true state s_t, only observations y_t:
```
y_t = C s_t + v_t,  where v_t ~ N(0, Σ_y)
```

### POMDP Framework

A POMDP is (S, O, A, P_{sa}, O(o|s), R):
- **O**: Observation space
- **O(o|s)**: Observation model

**Belief state**: Distribution P(s_t | y_1, ..., y_t)

For Gaussian models, belief is Gaussian: N(ŝ_{t|t}, Σ_{t|t})

### The LQG Strategy

**Step 1**: Estimate state using observations (Kalman filter)

**Step 2**: Use estimated state mean ŝ_{t|t} as if it were true state

**Step 3**: Apply LQR: a_t = -L_t ŝ_{t|t}

**Why it works**: LQR doesn't depend on noise, and ŝ is the best estimate.

### The Kalman Filter

Efficiently maintains belief state with:

**Predict step** (time update):
```
ŝ_{t+1|t} = A ŝ_{t|t}
Σ_{t+1|t} = A Σ_{t|t} A^T + Σ_s
```

**Update step** (measurement update):
```
K_t = Σ_{t+1|t} C^T (C Σ_{t+1|t} C^T + Σ_y)^{-1}  (Kalman gain)
ŝ_{t+1|t+1} = ŝ_{t+1|t} + K_t (y_{t+1} - C ŝ_{t+1|t})
Σ_{t+1|t+1} = Σ_{t+1|t} - K_t C Σ_{t+1|t}
```

**Key property**: Computation is O(1) per timestep (not O(t)!).

### Full LQG Algorithm

```
1. Run Kalman filter (forward pass) to compute K_t, Σ_{t|t}

2. Run LQR (backward pass) to compute L_t, Φ_t

3. Execute: a_t = -L_t ŝ_{t|t}
```

## Key Takeaways

1. **LQR** gives closed-form optimal control for linear-quadratic problems
2. **Optimal policy is linear**: a = -Ls (certainty equivalence)
3. **DDP** extends to nonlinear via iterative linearization
4. **LQG** handles partial observability via Kalman filtering
5. **Separation principle**: Estimation and control can be solved separately

## Practical Notes

- **LQR is widely used**: Drones, robots, self-driving cars
- **Model learning**: Can estimate A, B from data
- **iLQR**: Variant of DDP that only uses first-order dynamics approximation
- **MPC (Model Predictive Control)**: Replan at each step with updated state
- **Software**: MATLAB Control Toolbox, Python `control` library

