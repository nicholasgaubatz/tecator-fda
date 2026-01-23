import numpy as np
import os
from pathlib import Path
import pickle
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV

from tecatorfda.ols_models import (
    plot_and_save_ols_ridge_cv_boxplots,
)
from tecatorfda.data import load_tecator_fat


def main():
    # Load the fetched/processed data.
    tecator_df, fat_df, wavelength_grid, wavelength_unit = load_tecator_fat()
    
    # Be explicit in data shapes.
    X = tecator_df.to_numpy()
    y = fat_df.to_numpy().ravel()

    # Create a directory to save these results to.
    data_path = Path(__file__).parent.parent / "data" / "02_ridge"
    data_path.mkdir(parents=True, exist_ok=True)

    # If we don't already have 10-repeated 10-fold CV results saved, generate them.
    results_path = data_path / "repeated_cv_results"
    results_file = results_path / "results.pkl"
    if not results_file.exists():
        # Create this directory.
        results_path.mkdir(parents=True, exist_ok=True)

        # Define hyperparameters for fit.
        alphas = np.logspace(-10, 10, 201)

        # Inner CV: within each fold, standardize according to the train data, then perform CV for alpha on the train data.
        pipe = make_pipeline(
            StandardScaler(), RidgeCV(alphas=alphas, scoring="r2", cv=10)
        )

        # Outer CV: repeated 10 times, just like in the OLS script.
        outer_cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=0)
        scores = cross_val_score(
            pipe, X, y, cv=outer_cv, scoring="r2"
        )

        # Save all 100 scores.
        with open(results_file, "wb") as f:
            pickle.dump(scores, f)
    else:
        with open(results_file, "rb") as f:
            scores = pickle.load(f)

    ### Generate and save a CV boxplot comparison.

    plots_path = data_path / "plots"
    plots_path.mkdir(parents=True, exist_ok=True)

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

    plot_and_save_ols_ridge_cv_boxplots(
        ols_scores=ols_cv_scores,
        ridge_scores=scores,
        filepath=plots_path / "comparison.png",
    )


if __name__ == "__main__":
    main()
