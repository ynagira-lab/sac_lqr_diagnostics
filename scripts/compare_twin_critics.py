"""Compare saved SAC Twin Critics at 100k and 300k without retraining.

Place saved model ZIPs in the models/ directory.
Run: python scripts/compare_twin_critics.py
Outputs: figures/twin_critic_100k_vs_300k.png,
         figures/action_comparison.png, figures/action_summary.csv
"""
from pathlib import Path
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "figures"
OUTPUT.mkdir(exist_ok=True)
MODELS = {"100k": ROOT / "models" / "sac_lqr_trained_100k.zip",
          "300k": ROOT / "models" / "sac_lqr_trained_300k.zip"}
X_TEST = np.array([1.0, 0.0], dtype=np.float32)
U_GRID = np.linspace(-10.0, 10.0, 401)
U_LQR = -2.5853072593  # For the A, B, Q, R in experiment01e


def diagnose(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing saved model: {path}")
    model = SAC.load(str(path), device="cpu")
    obs = np.repeat(X_TEST[None, :], len(U_GRID), axis=0)
    physical_actions = U_GRID[:, None].astype(np.float32)
    scaled_actions = model.policy.scale_action(physical_actions).astype(np.float32)
    with torch.no_grad():
        q1_tensor, q2_tensor = model.critic(
            torch.as_tensor(obs, dtype=torch.float32, device=model.device),
            torch.as_tensor(scaled_actions, dtype=torch.float32, device=model.device),
        )
    q1 = q1_tensor.cpu().numpy().ravel()
    q2 = q2_tensor.cpu().numpy().ravel()
    qmin = np.minimum(q1, q2)
    actor, _ = model.predict(X_TEST, deterministic=True)
    return {"Q1": q1, "Q2": q2, "min-Q": qmin,
            "Actor": float(np.asarray(actor).ravel()[0]),
            "Q1 argmax": float(U_GRID[np.argmax(q1)]),
            "Q2 argmax": float(U_GRID[np.argmax(q2)]),
            "min-Q argmax": float(U_GRID[np.argmax(qmin)])}


def main():
    results = {label: diagnose(path) for label, path in MODELS.items()}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    for ax, (label, result) in zip(axes, results.items()):
        # Shift each landscape by its own peak; preserve Q1/Q2 differences.
        reference = np.max(result["min-Q"])
        for key in ("Q1", "Q2", "min-Q"):
            ax.plot(U_GRID, result[key] - reference,
                    linewidth=2 if key == "min-Q" else 1.5, label=key)
        ax.axvline(U_LQR, linestyle="--", color="black", label="LQR")
        ax.axvline(result["Actor"], linestyle=":", color="gray", label="Actor")
        ax.set_title(f"{label} steps")
        ax.set_xlabel("Physical action u")
        ax.grid(alpha=0.25)
        ax.set_xlim(-10, 5)
    axes[0].set_ylabel("Reward-Q minus max(min-Q) [within each model]")
    axes[0].legend()
    fig.suptitle("Twin Critic landscapes at x = [1, 0]")
    fig.tight_layout()
    fig.savefig(OUTPUT / "twin_critic_100k_vs_300k.png", dpi=200)
    plt.close(fig)

    labels = ["Q1 argmax", "Q2 argmax", "min-Q argmax", "Actor"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    width = 0.34
    for offset, step in [(-width / 2, "100k"), (width / 2, "300k")]:
        values = [results[step][key] for key in labels]
        ax.bar(x + offset, values, width, label=step)
    ax.axhline(U_LQR, linestyle="--", color="black", label="LQR optimum")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Physical action u")
    ax.set_title("Action diagnostics: 100k vs 300k")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT / "action_comparison.png", dpi=200)
    plt.close(fig)

    with (OUTPUT / "action_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["checkpoint", "LQR", *labels,
                         "actor_abs_error", "minq_abs_error"])
        for step, result in results.items():
            writer.writerow([step, U_LQR, *(result[key] for key in labels),
                             abs(result["Actor"] - U_LQR),
                             abs(result["min-Q argmax"] - U_LQR)])
    for step, result in results.items():
        print(f"{step}: " + ", ".join(f"{key}={result[key]:.3f}" for key in labels))
    print(f"Saved figures and CSV to {OUTPUT}")


if __name__ == "__main__":
    main()
