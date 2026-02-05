import argparse
from pathlib import Path
import pickle

import numpy as np
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor

from tecatorfda.all_models import (
    plot_and_save_cv_boxplots,
)
from tecatorfda.data import load_tecator_fat


def main():
    # Parse arguments.
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--data-location-directory", type=Path, required=True)
    p.add_argument("--artifacts-location-directory", type=Path, required=True)
    args = p.parse_args()

    # Load data.
    data_path = args.data_location_directory.expanduser().resolve()
    artifacts_path = args.artifacts_location_directory.expanduser().resolve()
    tecator_df, fat_df, wavelength_grid, wavelength_unit = load_tecator_fat(
        location=data_path
    )

    # Output directories.
    output_path = args.out_dir.expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    X = tecator_df.to_numpy()
    y = fat_df.to_numpy().ravel()

    results_path = output_path / "repeated_cv_results"
    results_file = results_path / "results.pkl"
    plots_path = output_path / "plots"

    if not results_file.exists():
        results_path.mkdir(parents=True, exist_ok=True)
        plots_path.mkdir(parents=True, exist_ok=True)

        # Define some basic random forest hyperparameters for a baseline comparison.
        rf = RandomForestRegressor(
            n_estimators=100,
            random_state=0,
        )

        # Define a small grid to search over.
        param_grid = {
             "max_features": ["sqrt", 0.3, 0.5],
             "min_samples_leaf": [1, 5, 10],
        }

        # Inner CV: performed once.
        inner_cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=0)

        # Outer CV: repeated 10 times, just like in the OLS script.
        outer_cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=0)
        
        # Specify the grid search.
        gs = GridSearchCV(
             rf, param_grid=param_grid, cv=inner_cv, scoring="r2"
        )
        
        scores = cross_val_score(
            gs, X, y, cv=outer_cv, scoring="r2"
        )

        # Save all 100 scores.
        with open(results_file, "wb") as f:
            pickle.dump(scores, f)
    else:
        with open(results_file, "rb") as f:
            scores = pickle.load(f)

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

    # Load the FLR CV scores.
    with open(
        str(
            (
                artifacts_path
                / "03_flr"
                / "repeated_cv_results"
                / "results.pkl"
            )
        ),
        "rb") as f:
            flr_scores = pickle.load(f)["r2_scores"]

    plot_and_save_cv_boxplots(
        all_scores=[ols_cv_scores, ridge_scores, flr_scores, scores],
        model_names=["OLS", "Ridge", "FLR", "RF"],
        filepath=plots_path / "comparison.png",
    )


if __name__ == "__main__":
    main()
