"""Local, shareable figures from one explicit evaluation output."""

from pathlib import Path

import numpy as np
import pandas as pd


def plot_evaluation(predictions, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.read_csv(predictions, dtype={"patient_id": str})
    if frame.split.nunique() != 1:
        raise ValueError("A figure must use one explicit split")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    grid = np.linspace(0, 30, 301)
    for i, (mode, part) in enumerate(frame.groupby("mode")):
        errors = part.error_px.to_numpy()
        cdf = np.array([(np.isfinite(errors) & (errors <= x)).sum() / len(errors) for x in grid])
        axes[0].step(grid, cdf, where="post", label=mode, linestyle="-" if i == 0 else "--", linewidth=2)
        average = part.groupby("vertebra").error_px.mean()
        axes[1].plot(average.index, average.values, marker="o", label=mode, linestyle="-" if i == 0 else "--")
    axes[0].set(
        xlabel="Euclidean threshold (original image px)",
        ylabel="Fraction of all visible GT points",
        ylim=(0, 1),
        title="Localization CDF (failures included)",
    )
    axes[1].set(
        xlabel="Vertebra (T1-T11)",
        ylabel="Mean Euclidean error (px)",
        title="Error by vertebra (successful predictions)",
    )
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle(
        f"{frame.split.iloc[0]} | {frame.image.nunique()} images | {frame.patient_id.nunique()} patients"
    )
    fig.savefig(out / "evaluation.png", dpi=180)
    fig.savefig(out / "evaluation.svg")
    plt.close(fig)
