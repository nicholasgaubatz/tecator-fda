import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tecatorfda.fda_models import (
    matrix_formation,
    coef_computation_and_metric_evaluation,
)

def sweep_fda_to_tidy_df(
    tecator_fdatagrid,
    y,
    KX_values,
    Kbeta_values,
    lam_values,
    *,
    cache_matrices=True,
    include_timing=True,
    verbose_progress=True,
):
    """
    Sweep over (K_X, K_beta, lam) and return a tidy (long) DataFrame.

    Assumes you have:
      - matrix_formation(tecator_fdatagrid, K_X=..., K_beta=...) -> (..., Z, R_0)
      - coef_computation_and_metric_evaluation(lam, Z, R_0, y, verbose=False)
        -> (zeta_hat, var_zeta, df, R2, R2_pred, CV_score)
    """
    y = np.asarray(y).ravel()
    KX_values = list(KX_values)
    Kbeta_values = list(Kbeta_values)
    lam_values = np.asarray(lam_values, dtype=float)
    log10_lams = np.log10(lam_values)

    # Cache for (Z, R_0) keyed by (KX, Kbeta)
    mat_cache = {} if cache_matrices else None

    rows = []
    total = len(KX_values) * len(Kbeta_values) * len(lam_values)
    done = 0
    t0_all = time.time()

    for KX in KX_values:
        for Kb in Kbeta_values:
            key = (KX, Kb)

            if cache_matrices and key in mat_cache:
                Z, R_0 = mat_cache[key]
            else:
                # NOTE: adjust unpacking if your matrix_formation returns different items.
                # You said: X_basis, X_basis_representation, beta_basis, J, Z, R_0 = matrix_formation(...)
                _, _, _, _, Z, R_0 = matrix_formation(tecator_fdatagrid, K_X=KX, K_beta=Kb)

                if cache_matrices:
                    mat_cache[key] = (Z, R_0)

            for lam, loglam in zip(lam_values, log10_lams):
                t0 = time.time()

                _, _, df, R2, R2_pred, CV_score = coef_computation_and_metric_evaluation(
                    lam, Z, R_0, y, verbose=False
                )

                row = {
                    "K_X": KX,
                    "K_beta": Kb,
                    "lam": lam,
                    "log10_lam": float(loglam),
                    "R2_pred": float(R2_pred),
                    "CV_score": float(CV_score),
                    "df": float(df),
                    "R2": float(R2),
                }

                if include_timing:
                    row["time_sec"] = time.time() - t0

                rows.append(row)

                done += 1
                if verbose_progress and (done % max(1, total // 10) == 0 or done == total):
                    elapsed = time.time() - t0_all
                    print(f"{done}/{total} runs ({done/total:.0%}) | elapsed {elapsed:.1f}s")

    df = pd.DataFrame(rows)

    # Helpful sorting for plotting/pivoting
    df = df.sort_values(["K_X", "K_beta", "log10_lam"]).reset_index(drop=True)
    return df

def save_small_multiples_heatmaps_from_df(df, value_col="R2_pred", cmap="seismic", filepath=None):
    KX_values = sorted(df["K_X"].unique())
    Kbeta_values = sorted(df["K_beta"].unique())
    log10_lams = sorted(df["log10_lam"].unique())

    n_panels = len(KX_values)
    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))

    vmin = df[value_col].min()
    vmax = df[value_col].max()

    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    last_im = None
    for idx, KX in enumerate(KX_values):
        ax = axes[idx]
        sub = df[df["K_X"] == KX]

        # Pivot to a matrix with rows = K_beta, cols = log10_lam
        M = (
            sub.pivot(index="K_beta", columns="log10_lam", values=value_col)
            .reindex(index=Kbeta_values, columns=log10_lams)
            .to_numpy()
        )

        last_im = ax.imshow(
            M,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=[min(log10_lams), max(log10_lams), min(Kbeta_values), max(Kbeta_values)],
        )
        ax.set_title(f"$K_X = {KX}$")
        ax.set_xlabel(r"$\log_{10}\lambda$")
        ax.set_ylabel(r"$K_\beta$")
        ax.set_yticks(Kbeta_values)

    for ax in axes[n_panels:]:
        ax.axis("off")

    fig.colorbar(last_im, ax=axes[:n_panels], shrink=0.9, label=value_col)
    
    # fig.tight_layout()
    fig.savefig(filepath, dpi=450) #, bbox_inches="tight")
    plt.close(fig)

def save_r2pred_vs_lambda_by_Kbeta(
    df,
    *,
    value_col="R2_pred",
    cmap="viridis",
    alpha=0.8,
    linewidth=1.5,
    filepath=None,
):
    """
    For each K_X, plot R2_pred vs log10(lambda),
    with one line per K_beta.
    """

    KX_values = sorted(df["K_X"].unique())
    Kbeta_values = sorted(df["K_beta"].unique())

    n_panels = len(KX_values)
    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.5 * ncols, 3.5 * nrows),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()

    # Colormap for K_beta lines
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(Kbeta_values)))

    for ax, KX in zip(axes, KX_values):
        sub_KX = df[df["K_X"] == KX]

        for color, Kb in zip(colors, Kbeta_values):
            sub = sub_KX[sub_KX["K_beta"] == Kb]

            if sub.empty:
                continue

            ax.plot(
                sub["log10_lam"],
                sub[value_col],
                marker="o",
                color=color,
                alpha=alpha,
                linewidth=linewidth,
                label=f"$K_\\beta={Kb}$",
            )

        ax.set_title(f"$K_X = {KX}$")
        ax.set_xlabel(r"$\log_{10}\lambda$")
        ax.set_ylabel(r"Predictive $R^2$")

    # Turn off unused axes
    for ax in axes[len(KX_values):]:
        ax.axis("off")

    # One shared legend (outside the grid)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center right",
        bbox_to_anchor=(1.02, 0.5),
        title=r"$K_\beta$",
    )

    # fig.tight_layout()
    fig.savefig(filepath, dpi=450)#, bbox_inches="tight")
    plt.close(fig)

def save_heatmap_max_r2pred(df, *, value_col="R2_pred", cmap="seismic", filepath=None):
    """
    Heatmap of max over lambda of predictive R^2 for each (K_X, K_beta).
    Assumes df has columns: K_X, K_beta, value_col.
    """
    # 1) Reduce over lambda: max R2_pred for each (K_X, K_beta)
    df_best = (
        df.groupby(["K_X", "K_beta"], as_index=False)[value_col]
          .max()
          .rename(columns={value_col: f"max_{value_col}"})
    )

    # Get all the columns for the best lambda for each K.
    idx = df.groupby(["K_X", "K_beta"])[value_col].idxmax()
    best_lambda_table = df.loc[idx].sort_values(["K_X", "K_beta"]).reset_index(drop=True)

    # 2) Sort axes
    KX_values = sorted(df_best["K_X"].unique())
    Kbeta_values = sorted(df_best["K_beta"].unique())

    # 3) Pivot to matrix with rows=K_beta, cols=K_X
    M = (
        df_best.pivot(index="K_beta", columns="K_X", values=f"max_{value_col}")
              .reindex(index=Kbeta_values, columns=KX_values)
              .to_numpy()
    )

    # 4) Plot
    vmin = np.nanmin(M)
    vmax = np.nanmax(M)

    fig, ax = plt.subplots(figsize=(1.1 * len(KX_values) + 2, 0.5 * len(Kbeta_values) + 2))

    im = ax.imshow(
        M,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[min(KX_values), max(KX_values), min(Kbeta_values), max(Kbeta_values)],
    )

    ax.set_xlabel(r"$K_X$")
    ax.set_ylabel(r"$K_\beta$")
    ax.set_title(rf"$\max_\lambda$ Predictive $R^2$ ({value_col})")

    ax.set_xticks(KX_values)
    ax.set_yticks(Kbeta_values)

    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label(r"$\max_\lambda R^2_{\mathrm{pred}}$")

    fig.tight_layout()
    fig.savefig(filepath, dpi=450, bbox_inches="tight")
    plt.close(fig)

    return best_lambda_table  # handy to inspect / save

def save_r2pred_vs_df(
    df,
    *,
    color_by="K_X",
    alpha=0.6,
    s=35,
    cmap="viridis",
    filepath=None,
):
    """
    Scatter plot of predictive R^2 vs effective df,
    colored by K_X or K_beta.
    """

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    values = sorted(df[color_by].unique())
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(values)))

    for val, color in zip(values, colors):
        sub = df[df[color_by] == val]
        ax.scatter(
            sub["df"],
            sub["R2_pred"],
            s=s,
            alpha=alpha,
            color=color,
            label=f"{color_by}={val}",
        )

    ax.set_xlabel("Effective degrees of freedom")
    ax.set_ylabel(r"Predictive $R^2$")
    ax.set_title(r"Predictive $R^2$ vs effective degrees of freedom")

    ax.legend(title=color_by, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(filepath, dpi=450, bbox_inches="tight")
    plt.close(fig)

def save_binned_r2pred_vs_df(
    df,
    *,
    n_bins=20,
    filepath=None,
):
    """
    Bin df and plot mean ± SD of predictive R^2 in each bin.
    """

    df = df.copy()

    # Create df bins
    df["df_bin"] = pd.cut(df["df"], bins=n_bins)

    grouped = (
        df.groupby("df_bin", observed=True)["R2_pred"]
          .agg(["mean", "std", "count"])
          .reset_index()
    )

    # Bin centers for plotting
    bin_centers = grouped["df_bin"].apply(lambda x: x.mid)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    ax.plot(bin_centers, grouped["mean"], marker="o", label="Mean")
    ax.fill_between(
        bin_centers,
        grouped["mean"] - grouped["std"],
        grouped["mean"] + grouped["std"],
        alpha=0.25,
        label="±1 SD",
    )

    ax.set_xlabel("Effective degrees of freedom")
    ax.set_ylabel(r"Predictive $R^2$")
    ax.set_title(r"Predictive $R^2$ vs df (binned)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(filepath, dpi=450, bbox_inches="tight")
    plt.close(fig)

