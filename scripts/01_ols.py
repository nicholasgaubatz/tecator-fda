import argparse
from pathlib import Path
import pickle

from tecatorfda.ols_models import (
    perform_ols,
    perform_ols_cv,
    perform_ols_holdout,
    plot_and_save_diagnostics,
)
from tecatorfda.data import load_tecator_fat


def main():
    # Parse the arguments to the script.
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--data-location-directory", type=Path, required=True)
    args = p.parse_args()
    
    # Load the fetched/processed data.
    data_path = args.data_location_directory.expanduser().resolve()
    tecator_df, fat_df, wavelength_grid, wavelength_unit = load_tecator_fat(location=data_path)

    # Create a directory to save these results to from the arguments.
    output_path = args.out_dir.expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    # Compute OLS on the entire data set.
    results_full = perform_ols(tecator_df.values, fat_df.values)

    # Compute OLS for random_state=0.
    train_results_0, test_results_0 = perform_ols_holdout(
        tecator_df.values, fat_df.values, random_state=0
    )

    # Compute OLS for random_state=42.
    train_results_42, test_results_42 = perform_ols_holdout(
        tecator_df.values, fat_df.values, random_state=42
    )

    ### Save all the results to data/01_ols/results.

    results_path = output_path / "results"
    results_path.mkdir(parents=True, exist_ok=True)

    all_results = {
        "ols_full": results_full,
        "ols_0_train": train_results_0,
        "ols_0_test": test_results_0,
        "ols_42_train": train_results_42,
        "ols_42_test": test_results_42,
    }

    for key, value in all_results.items():
        with open(results_path / (key + ".pkl"), "wb") as f:
            pickle.dump(value, f)

    ### Save some important plots to data/01_ols/plots.

    plots_path = output_path / "plots"
    plots_path.mkdir(parents=True, exist_ok=True)

    for key in all_results.keys():
        plot_and_save_diagnostics(all_results[key], plots_path / (key + ".png"))

    ### Run OLS repeated CV and save.

    results_cv = perform_ols_cv(tecator_df.values, fat_df.values, random_state=0)

    with open(results_path / "ols_cv.pkl", "wb") as f:
        pickle.dump(results_cv, f)


if __name__ == "__main__":
    main()
