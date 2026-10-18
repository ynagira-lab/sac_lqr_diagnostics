# Diagnosing SAC Learning Against an Analytical LQR Reference — Comparing the Actor and Twin Critics

[日本語版 / Japanese version](README_ja.md)

## 1. Background and Motivation

For a linear quadratic regulator (LQR), the optimal state-feedback law can be computed from the Riccati equation when the system model and quadratic cost function are known. In reinforcement learning, by contrast, a policy is learned through interaction with an environment without providing the optimal control law in advance.

This experiment asks a simple question: **when Soft Actor-Critic (SAC) is trained on a control problem with a known analytical LQR solution, how closely does its learned action approach the LQR reference?**

We trained SAC and saved models after 100,000 (100k) and 300,000 (300k) environment steps. At 100k, the SAC action differed substantially from the LQR reference. By 300k, the difference had become much smaller. To examine this change, we compared not only the learned Actor's output but also SAC's two learned Critics (Twin Critics) and their pointwise minimum (min-Q).

This study addresses two questions:

1. Does the deterministic SAC Actor move closer to the analytical LQR action as training progresses?
2. How do changes in the Actor relate to the shapes of the two Critic functions and min-Q?

The goal is not to establish that SAC and LQR solve mathematically identical optimization problems, nor to prove general convergence. Rather, **this is a case study of learned policies and value estimates on a control problem with a computable optimal reference.**

## 2. Control Problem and Analytical LQR Solution

Consider the following discrete-time linear system with two states and one control input, where $x_k\in\mathbb{R}^2$ and $u_k\in\mathbb{R}$:

$$
x_{k+1}=Ax_k+Bu_k,\qquad
A=\begin{bmatrix}1&0.1\\0&1\end{bmatrix},\qquad
B=\begin{bmatrix}0\\0.1\end{bmatrix}.
$$

The matrix $A$ has a repeated eigenvalue at 1, so the open-loop system is not asymptotically stable. We consider stabilization through state feedback.

The stage cost is

$$
\ell(x,u)=x^\top Qx+u^\top Ru,\qquad Q=I_2,\quad R=0.1.
$$

The undiscounted, infinite-horizon LQR objective is

$$
J(x_0)=\sum_{k=0}^{\infty}\ell(x_k,u_k).
$$

Using the stabilizing solution $P$ of the discrete-time algebraic Riccati equation, the optimal feedback law is

$$
K=(R+B^\top PB)^{-1}B^\top PA,\qquad u_{\mathrm{LQR}}(x)=-Kx.
$$

For the action comparison, we fix the evaluation state at **$x_{\mathrm{test}}=[1,0]^\top$**. The analytical LQR action at this state is

$$
u_{\mathrm{LQR}}(x_{\mathrm{test}})\simeq -2.5853.
$$

**We use this value as the reference against which to compare the SAC Actor output and the actions preferred by the learned Critics.**

For reference, the optimal LQR *cost* action-value function can be written as

$$
Q^*_{\mathrm{LQR,cost}}(x,u)
=\ell(x,u)+(Ax+Bu)^\top P(Ax+Bu).
$$

This function is **minimized**. By contrast, the SAC Critics considered below estimate action values in terms of *reward*, for which **larger** values are generally preferred. We therefore do not compare the numerical values of these two Q-functions directly. Instead, we primarily compare the **physical control actions $u$** selected or preferred by each method.

## 3. Overview of SAC

Soft Actor-Critic (SAC) is a reinforcement learning algorithm that uses an Actor to learn a policy and Critics to estimate state-action values.

- **Actor** $\pi_\theta(u\mid x)$: represents a distribution over actions $u$ given a state $x$. For this study, we obtain a deterministic action from the trained Actor.
- **Critic** $Q_{\phi_i}(x,u)$: estimates the value of choosing action $u$ at state $x$ in terms of future rewards.
- **Twin Critics**: standard SAC trains two Critics, $Q_1$ and $Q_2$, and uses the smaller estimate during updates to mitigate overestimation. The use of two Critics is a standard feature of SAC, not a modification introduced in this experiment.

We define the pointwise minimum of the two Critics as

$$
Q_{\min}(x,u)=\min\{Q_1(x,u),Q_2(x,u)\}.
$$

Ensemble-based approaches with three or more Critics also exist, but this experiment focuses on the standard two-Critic configuration.

Importantly, **the Actor's output is not obtained by simply computing $\arg\max_u Q_{\min}(x,u)$**. SAC updates a parameterized stochastic policy using an entropy-regularized objective. Consequently, the deterministic Actor action need not coincide with the maximizer of min-Q.

The following diagram illustrates the relationships among the components examined in this study. The Actor generates an action from a state; the two Critics separately estimate Q-values from a state-action pair; and min-Q selects the smaller estimate. **The analytical LQR solution is not used to train SAC; it is computed separately for evaluation.** The arrows illustrate the main information flow rather than every update equation in SAC.

![Relationship between SAC components and the LQR reference](figures/sac_lqr_overview.svg)

We use SAC from Stable-Baselines3 (SB3). Both the policy and value functions are approximated using multilayer perceptrons (MLPs), i.e., neural networks composed of fully connected layers.

## 4. Experimental Setup and Diagnostic Method

### 4.1 Training Configuration

| Parameter | Setting |
|---|---|
| Algorithm | SB3 SAC |
| Network | Two-layer MLP, 64 units per layer |
| Learning rate | $10^{-4}$ |
| Replay buffer capacity | 100,000 |
| Learning starts | 5,000 steps |
| Batch size | 256 |
| Discount factor $\gamma$ | 0.99 |
| Target update coefficient $\tau$ | 0.005 |
| Entropy coefficient | Automatically adjusted |
| Physical action range | $[-10,10]$ |
| Initial state | Uniformly sampled from $[-2,2]^2$ |
| Episode termination | Truncated at 200 steps, or terminated if the state norm exceeds 10 |
| Random seed | 0 |

We saved the **100k** and **300k** checkpoints during a single continuous training run. Thus, the two results represent **different stages of the same training trajectory**, rather than independently trained models.

### 4.2 Why Examine the Critics?

When the Actor's action differs from the LQR reference, the Actor output alone does not reveal how the learned value estimates behave. We therefore fix $x_{\mathrm{test}}=[1,0]^\top$ and evaluate $Q_1(x_{\mathrm{test}},u)$, $Q_2(x_{\mathrm{test}},u)$, and $Q_{\min}(x_{\mathrm{test}},u)$ over 401 physical action values from $-10$ to $10$.

Fixing the state lets us visualize **which actions receive higher estimated values at the same state** as one-dimensional curves. We compare the maximizer of each curve with the Actor action and the LQR reference to examine how their relationships change during training.

Before passing actions to the SB3 Critics, physical actions in $[-10,10]$ are transformed to the normalized action coordinates expected by the networks. See `scripts/compare_twin_critics.py` for the implementation.

## 5. Results and Discussion

### 5.1 The Actor Approached the LQR Reference from 100k to 300k

| Diagnostic metric (physical action $u$) | 100k | 300k |
|---|---:|---:|
| Analytical LQR action | -2.585 | -2.585 |
| $\arg\max_u Q_1(x_{\mathrm{test}},u)$ | -2.550 | -0.700 |
| $\arg\max_u Q_2(x_{\mathrm{test}},u)$ | -8.000 | -2.700 |
| $\arg\max_u Q_{\min}(x_{\mathrm{test}},u)$ | -6.050 | -2.700 |
| Deterministic SAC Actor | -4.897 | -2.842 |
| Absolute Actor–LQR action error | 2.312 | 0.257 |
| Absolute min-Q-maximizer–LQR action error | 3.465 | 0.115 |

The first result to examine is the **Actor action**. At 100k, the Actor outputs $-4.897$, substantially different from the LQR reference of $-2.585$. At 300k, the Actor outputs $-2.842$, reducing the absolute error from **2.312 to 0.257—approximately an 89% reduction**. At least at this evaluation state, the Actor moved substantially closer to the LQR reference as training progressed.

Next, consider the **Critic maximizers**. At 100k, only the maximizer of $Q_1$ is close to the LQR action, while those of $Q_2$ and min-Q are far away. At 300k, the maximizers of $Q_2$ and min-Q move close to the LQR reference, even though the maximizer of $Q_1$ moves farther away.

### 5.2 Reading the Twin Critic Curves

![Twin Critic curves at 100k and 300k](figures/twin_critic_100k_vs_300k.png)

The estimated action value $Q_i(x,u)$ is a neural-network approximation of the value of selecting action $u$ at state $x$ and then following the learned policy, in terms of **future discounted rewards (with SAC's entropy terms taken into account)**. It is not the same as the analytical LQR cost Q-function. For a fixed trained Critic, the estimate is uniquely determined by the specified state and action.

In the figure, the **horizontal axis is the physical action $u$**, and the **vertical axis is the estimated Critic Q-value after subtracting a display offset $c$**. For each checkpoint, we define

$$
c=\max_{u\in\mathcal U_{\mathrm{grid}}} Q_{\min}(x_{\mathrm{test}},u),\qquad
\widetilde Q_i(x_{\mathrm{test}},u)=Q_i(x_{\mathrm{test}},u)-c.
$$

The same offset $c$ is subtracted from all three curves, $Q_1$, $Q_2$, and $Q_{\min}$. In other words, **we shift the curves vertically so that the maximum of min-Q appears at zero**. This changes neither the learned Q-values themselves nor the actions at which the curves attain their maxima. Because the offset is calculated separately at 100k and 300k, the absolute levels of the Q-values cannot be compared directly across the two panels.

The curves are shown together with vertical markers for the LQR and Actor actions:

- **100k (left):** The peak of $Q_1$ lies near the LQR reference, but the maximizer of $Q_2$ is shifted substantially to the left. The maximizer of the pointwise minimum, min-Q, is also far from the LQR action.
- **300k (right):** The maximizers of $Q_2$ and min-Q are near the LQR reference, and the Actor marker also moves closer to the LQR marker. The maximizer of $Q_1$, however, moves away from the LQR reference.

### 5.3 Reading the Action Comparison

![Comparison of LQR, Critic-maximizing, and Actor actions](figures/action_comparison.png)

This figure compares only the **locations of the selected or preferred actions** at each checkpoint. **The closer a value is to the black dashed line marking the LQR action, the smaller its action error under the metric used here** (although this does not imply optimality under SAC's own objective). At 100k, the Actor and min-Q maximizer are far from the LQR reference; at 300k, both are much closer.

A notable observation is that **the Actor's action error decreases even though the maximizer of $Q_1$ alone moves farther from the LQR reference**. Thus, improvement of every individual Critic maximizer toward the LQR action is not a prerequisite for the observed improvement in the Actor's action.

However, **we cannot conclude that the change in min-Q caused the Actor's improvement**. The Actor is updated through entropy-regularized policy learning rather than direct maximization of min-Q at a single state. This experiment documents how the Actor and the action preferences of the Critics change along the same training trajectory; it does not isolate which individual update mechanism produced the Actor's improvement.

## 6. Reproducing the Results

This repository separates the training script from the diagnostic script that generates plots using saved models.

Install NumPy, SciPy, Gymnasium, Stable-Baselines3, PyTorch, and Matplotlib in your Python environment. See `requirements.txt` for dependency details.

From the repository root, run training:

```bash
python scripts/train_sac_lqr.py
```

This saves `models/sac_lqr_trained_100k.zip` and `models/sac_lqr_trained_300k.zip` from the same training run. If files with these names already exist, check the output location before running to avoid accidental overwriting. **If the saved models are already available, you can skip training.**

Then run the diagnostics:

```bash
python scripts/compare_twin_critics.py
```

The script generates:

- `figures/twin_critic_100k_vs_300k.png`
- `figures/action_comparison.png`
- `figures/action_summary.csv`

Retraining may not reproduce exactly the same numerical results if the environment or library versions differ. To reproduce the figures for the reported checkpoints, use the corresponding saved model files.

## 7. Conclusion

We examined the Actor and Twin Critics during SAC training on a linear quadratic control problem with a known analytical LQR action.

**The main observation** is that, at $x=[1,0]^\top$, the Actor action differs substantially from the LQR reference at 100k but approaches it closely by 300k. Meanwhile, the maximizers of the two individual Critics do not uniformly approach the LQR reference; they evolve differently from one another and from min-Q and the Actor.

These results illustrate the value of visualizing **both individual Critics and min-Q alongside the Actor**, rather than inspecting the final Actor output alone. They do not establish that changes in min-Q caused the Actor's improvement, nor do they demonstrate general convergence of SAC.

### Scope and Limitations of the Comparison

LQR minimizes an undiscounted infinite-horizon cost, whereas SAC in this experiment maximizes an entropy-regularized reward objective with discount factor $\gamma=0.99$ and finite episode boundaries. Therefore, **we do not require the SAC Q-values to match the optimal LQR cost Q-function**. Instead, we measure **how closely the Actor's physical action approaches the known LQR action** on the same plant. This provides a useful reference comparison, but a small action difference alone does not establish optimality under SAC's own objective.

The results also concern a single evaluation state and a single random seed. They do not establish action accuracy across the entire state space, closed-loop stability, or overall control performance.

## 8. Future Work

A natural next step is to extend the diagnostics beyond one fixed state: examine differences between Actor and Critic action preferences across the state space, test reproducibility across random seeds, and relate action differences to closed-loop cost and stability. This would allow a more systematic assessment of **how agreement with an analytical action reference relates to actual control performance**.

A broader direction is to compare and integrate model-based optimal control with learning-based control. Possible extensions include moving toward constrained control through comparisons among LQR, MPC, and iLQR, and exploring **how learning-driven performance improvements can be combined with control-theoretic stability conditions** through Residual RL and Lyapunov-based methods.
