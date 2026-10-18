"""Train SAC on the original 2-state LQR environment and save checkpoints.

Run from the repository root:
    python scripts/train_sac_lqr.py
    python scripts/train_sac_lqr.py --save-replay

The training environment and SAC hyperparameters follow experiment01e.
Unlike the legacy online monitor, this script does not evaluate a Critic using
unscaled physical actions. For the corrected diagnostic, run
scripts/compare_twin_critics.py after training.

Note: 100k and 300k checkpoints produced here belong to ONE training run.
Previously archived checkpoints may have been produced by separate runs;
therefore exact numerical agreement with historical results is not guaranteed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.linalg import solve_discrete_are
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

A = np.array([[1.0, 0.1], [0.0, 1.0]], dtype=np.float32)
B = np.array([[0.0], [0.1]], dtype=np.float32)
Q_COST = np.eye(2, dtype=np.float32)
R_COST = np.array([[0.1]], dtype=np.float32)


class LinearLQREnv(gym.Env):
    """Match the dynamics, reward, resets and stopping rules of experiment01e."""

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(2,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-10.0, high=10.0, shape=(1,), dtype=np.float32
        )
        self.max_steps = 200
        self.state_limit = 10.0
        self.state = np.zeros(2, dtype=np.float32)
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(
            low=-2.0, high=2.0, size=(2,)
        ).astype(np.float32)
        self.steps = 0
        return self.state.copy(), {}

    def step(self, action):
        u = np.clip(
            np.asarray(action, dtype=np.float32),
            self.action_space.low,
            self.action_space.high,
        ).reshape(1, 1)
        x = self.state.reshape(2, 1)
        cost = float((x.T @ Q_COST @ x + u.T @ R_COST @ u).item())
        next_state = (A @ x + B @ u).flatten()
        self.state = next_state.astype(np.float32)
        self.steps += 1
        terminated = bool(np.linalg.norm(next_state) > self.state_limit)
        truncated = bool(self.steps >= self.max_steps)
        return self.state.copy(), -cost, terminated, truncated, {}


class CheckpointCallback(BaseCallback):
    """Save model at the requested environment-step counts, without retraining."""

    def __init__(self, checkpoints, save_replay=False):
        super().__init__()
        self.checkpoints = set(checkpoints)
        self.save_replay = save_replay

    def _on_step(self):
        step = self.num_timesteps
        if step in self.checkpoints:
            label = f"{step // 1000}k"
            model_path = MODELS_DIR / f"sac_lqr_trained_{label}"
            self.model.save(str(model_path))
            print(f"Saved model: {model_path}.zip", flush=True)
            if self.save_replay:
                buffer_path = MODELS_DIR / f"sac_lqr_replay_buffer_{label}.pkl"
                self.model.save_replay_buffer(str(buffer_path))
                print(f"Saved replay buffer: {buffer_path}", flush=True)
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-replay", action="store_true")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow overwriting existing checkpoint files")
    args = parser.parse_args()
    if args.steps < 100_000:
        parser.error("--steps must be at least 100000 to save the 100k checkpoint")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoints = [step for step in (100_000, 300_000) if step <= args.steps]
    planned = [MODELS_DIR / f"sac_lqr_trained_{step // 1000}k.zip" for step in checkpoints]
    if args.save_replay:
        planned += [MODELS_DIR / f"sac_lqr_replay_buffer_{step // 1000}k.pkl" for step in checkpoints]
    if args.steps not in checkpoints:
        label = f"{args.steps // 1000}k" if args.steps % 1000 == 0 else str(args.steps)
        planned.append(MODELS_DIR / f"sac_lqr_trained_{label}.zip")
    existing = [path for path in planned if path.exists()]
    if existing and not args.overwrite:
        parser.error("Existing output file(s) would be overwritten: " +
                     ", ".join(str(path) for path in existing) +
                     ". Use an empty models/ directory or pass --overwrite explicitly.")
    np.random.seed(args.seed)

    p = solve_discrete_are(A, B, Q_COST, R_COST)
    k = np.linalg.solve(R_COST + B.T @ p @ B, B.T @ p @ A)
    print("LQR K =", k, "LQR action at x=[1,0] =", float(-(k @ [[1.0], [0.0]]).item()))

    env = LinearLQREnv()
    model = SAC(
        "MlpPolicy", env,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        ent_coef="auto",
        policy_kwargs=dict(net_arch=[64, 64]),
        verbose=1,
        seed=args.seed,
    )
    callback = CheckpointCallback(checkpoints, save_replay=args.save_replay)
    model.learn(total_timesteps=args.steps, callback=callback)
    if args.steps not in checkpoints:
        label = f"{args.steps // 1000}k" if args.steps % 1000 == 0 else str(args.steps)
        model.save(str(MODELS_DIR / f"sac_lqr_trained_{label}"))
    env.close()
    print("Training complete. Run: python scripts/compare_twin_critics.py")


if __name__ == "__main__":
    main()
