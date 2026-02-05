import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from pathlib import Path


def plot_and_save_cv_boxplots(
    all_scores: list[np.ndarray],
    model_names: list[str],
    filepath: Path,
):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    ax.boxplot(all_scores, tick_labels=model_names)
    ax.set_title("Repeated CV $R^{2}$ scores")
    ax.set_ylabel("$R^{2}$")
    ax.yaxis.set_major_locator(MultipleLocator(0.05)) # Set tick values 0.05 apart.
    ax.grid(
        axis="y",
    )

    fig.tight_layout()
    fig.savefig(filepath, dpi=450, bbox_inches="tight")
    plt.close(fig)
