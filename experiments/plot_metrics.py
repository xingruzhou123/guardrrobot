# experiments/plot_metrics.py
"""
Visualize results from the humanoid guardrail experiments.
Generates a bar chart of violation counts and safe success rate.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

LOG_PATH = Path("experiments/logs/results.csv")
OUT_PATH = Path("experiments/logs/metrics.png")

def main():
    if not LOG_PATH.exists():
        raise FileNotFoundError(f"No log file found at {LOG_PATH}")

    df = pd.read_csv(LOG_PATH)

    # --- Compute summary metrics ---
    df["violation_count"] = df["violations"].apply(lambda x: 0 if x == "[]" else len(str(x).split(",")))
    unsafe_rate = 100 * (df["safe_success"] == False).mean()
    mean_runtime = df["runtime_ms"].mean()

    print("=== Experiment Summary ===")
    print(df[["task", "violation_count", "safe_success", "runtime_ms"]])
    print(f"\nUnsafe rate: {unsafe_rate:.1f}%")
    print(f"Average guard runtime: {mean_runtime:.2f} ms")

    # --- Plot: Violations per Task ---
    plt.figure(figsize=(6, 4))
    plt.bar(df["task"], df["violation_count"], label="Violations", alpha=0.7)
    plt.ylabel("Violation count")
    plt.title("Guardrail Violations per Task")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(OUT_PATH)
    plt.show()

    print(f"\nPlot saved to {OUT_PATH.resolve()}")

if __name__ == "__main__":
    main()
