import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import pickle

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
from tecatorfda.fda_models import (
    repeated_cv_flr,
    plot_and_save_ols_ridge_flr_cv_boxplots,
)
from tecatorfda.ols_models import (
    plot_and_save_ols_ridge_cv_boxplots,
)

def main():
    # Load the fetched/processed data.
    _, fat_df, _, _ = load_tecator_fat()
    tecator_fdatagrid, _ = skfda.datasets.fetch_tecator(return_X_y=True, as_frame=False)

    # Be explicit in data shapes.
    fat = fat_df.values.flatten()

    # Create a directory to save these results to.
    data_path = Path(__file__).parent.parent / "data" / "03_flr"
    data_path.mkdir(parents=True, exist_ok=True)

    # If we don't already have 10-repeated 10-fold CV results saved, generate them.
    results_path = data_path / "repeated_cv_results"
    results_file = results_path / "results.pkl"
    if not results_file.exists():
        # Create this directory.
        results_path.mkdir(parents=True, exist_ok=True)

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

    plots_path = data_path / "plots"
    plots_path.mkdir(parents=True, exist_ok=True)

    # Load the OLS CV scores.
    with open(
        str(
            (
                Path(__file__).parent.parent
                / "data"
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
                Path(__file__).parent.parent
                / "data"
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
