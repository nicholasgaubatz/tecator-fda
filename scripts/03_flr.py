import argparse
import os
from pathlib import Path
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import skfda
from skfda.misc.operators import LinearDifferentialOperator
from skfda.misc.regularization import L2Regularization
from skfda.ml.regression import LinearRegression as FLinearRegression
from skfda.representation.basis import BSplineBasis
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV

from tecatorfda.data import load_tecator_fat
from tecatorfda.fda_analysis import (
    save_binned_r2pred_vs_df,
    save_heatmap_max_r2pred,
    save_r2pred_vs_df,
    save_r2pred_vs_lambda_by_Kbeta,
    save_small_multiples_heatmaps_from_df,
    sweep_fda_to_tidy_df,
)
from tecatorfda.fda_models import (
    repeated_cv_flr,
    plot_and_save_ols_ridge_flr_cv_boxplots,
)

def main():
    # Parse the arguments to the script.
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--data-location-directory", type=Path, required=True)
    p.add_argument("--artifacts-location-directory", type=Path, required=True)
    args = p.parse_args()
    
    # Load the fetched/processed data.
    data_path = args.data_location_directory.expanduser().resolve()
    artifacts_path = args.artifacts_location_directory.expanduser().resolve()
    tecator_df, fat_df, wavelength_grid, wavelength_unit = load_tecator_fat(location=data_path)

    # Create a directory to save these results to from the arguments.
    output_path = args.out_dir.expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    # Put the Tecator DataFrame in FDataGrid format.
    # tecator_fdatagrid, _ = skfda.datasets.fetch_tecator(return_X_y=True, as_frame=False)
    tecator_fdatagrid = skfda.FDataGrid(data_matrix=tecator_df.to_numpy(dtype=float), grid_points=wavelength_grid)

    # Be explicit in data shapes.
    fat = fat_df.values.flatten()


    # Define the paths we use throughout.
    analysis_path = output_path / "analysis_results"
    results_path = output_path / "repeated_cv_results"
    results_file = results_path / "results.pkl"
    plots_path = output_path / "plots"

    # If we don't already have results saved, generate them.
    if not results_file.exists():
        # Create this directory.
        analysis_path.mkdir(parents=True, exist_ok=True)
        results_path.mkdir(parents=True, exist_ok=True)

        # Create a plots directory.
        plots_path.mkdir(parents=True, exist_ok=True)

        ### Basis representation with smoothing parameter analysis.

        # User-defined parameters.
        KX_values = np.arange(5, 50, 5)
        Kbeta_values = np.arange(5, 50, 5)
        log10_lams = np.arange(-15, 5, 1.)

        # Consequences.
        lam_values = 10 ** log10_lams
        
        # Perform a hyperparameter sweep: K_X, K_beta, lambda.
        tidy_df = sweep_fda_to_tidy_df(
            tecator_fdatagrid=tecator_fdatagrid,
            y=fat,
            KX_values=KX_values,
            Kbeta_values=Kbeta_values,
            lam_values=lam_values,
            cache_matrices=True,
            include_timing=True,
            verbose_progress=True,
        )

        # Save this tidy_df.
        tidy_df.to_csv(analysis_path / "hyperparameter_sweep.csv", index=False)

        # Plot heatmaps for predictive R^2: each plot highlights a fixed K_X, and we
        # color the 2-dimensional (K_beta, log_lambda) grid.
        save_small_multiples_heatmaps_from_df(tidy_df, value_col="R2_pred", cmap="seismic", filepath=plots_path / "R2_heatmaps.png")

        # Plot predictive R^2 for each K_X based on K_beta and lambda.
        save_r2pred_vs_lambda_by_Kbeta(
            tidy_df,
            value_col="R2_pred",
            cmap="plasma",   # try: "viridis", "plasma", "tab10"
            filepath=plots_path / "R2_vs_lambda_per_K.png"
        )

        # Plot maximum predictive R^2 given K_X and K_beta.
        best_R2_df = save_heatmap_max_r2pred(tidy_df, value_col="R2_pred", cmap="seismic", filepath=plots_path / "best_R2.png")

        # Save this optimization DataFrame.
        best_R2_df.to_csv(analysis_path / "optimized_per_K.csv", index=False)

        # Plot predictive R^2 vs. df (what ChatGPT considers the coolest info).
        save_r2pred_vs_df(tidy_df, color_by="K_beta", filepath=plots_path / "R2_vs_df.png")

        # Plot again, but this time just means and standard deviations based on 15 lambda bins.
        save_binned_r2pred_vs_df(tidy_df, n_bins=15, filepath=plots_path / "R2_vs_df_binned.png")

        # Print progress.
        print("Done with analysis plot generation! Proceding to repeated CV.")

        ### Repeated CV evaluation.
        
        # Define hyperparameters for fit.
        lam_grid = 10 ** np.arange(-16, -6, 1.)
        K_X = 30
        K_beta = 30

        # Perform the cross-validations.
        flr_scores, chosen_lams = repeated_cv_flr(
            tecator_fdatagrid=tecator_fdatagrid,
            y=fat,
            K_X=K_X,
            K_beta=K_beta,
            lam_grid=lam_grid,
            random_state=0,
        )

        # Save all 100 scores and chosen hyperparameter values for each.
        with open(results_file, "wb") as f:
            pickle.dump(
                {"r2_scores": flr_scores, "chosen_lams": chosen_lams, "K_X": K_X, "K_beta": K_beta, "lam_grid": lam_grid},
                f,
            )
    else:
        with open(results_file, "rb") as f:
            obj = pickle.load(f)
        flr_scores = obj["r2_scores"]

    ### Generate and save a CV boxplot comparison.

    # Load the OLS CV scores.
    with open(
        str(
            (
                artifacts_path
                / "01_ols"
                / "results"
                / "ols_cv.pkl"
            )
        ),
        "rb",
    ) as f:
        ols_cv_scores = pickle.load(f).r2_scores

    # Load the Ridge CV scores.
    with open(
        str(
            (
                artifacts_path
                / "02_ridge"
                / "repeated_cv_results"
                / "results.pkl"
            )
        ),
        "rb") as f:
            ridge_scores = pickle.load(f)

    plot_and_save_ols_ridge_flr_cv_boxplots(
        ols_scores=ols_cv_scores,
        ridge_scores=ridge_scores,
        flr_scores=flr_scores,
        filepath=plots_path / "comparison.png",
    )


if __name__ == "__main__":
    main()
