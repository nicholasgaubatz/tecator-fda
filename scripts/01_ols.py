from pathlib import Path
import pickle

from tecatorfda.ols_models import (
    perform_ols,
    perform_ols_holdout,
    plot_and_save_diagnostics,
)
from tecatorfda.data import load_tecator_fat


def main():
    # Load the fetched/processed data.
    tecator_df, fat_df, wavelength_grid, wavelength_unit = load_tecator_fat()

    # Create a directory to save these results to.
    data_path = Path(__file__).parent.parent / "data" / "01_ols"
    data_path.mkdir(parents=True, exist_ok=True)

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

    results_path = data_path / "results"
    results_path.mkdir(parents=True, exist_ok=True)

    all_results = {
        "ols_full": results_full,
        "ols_0_train": train_results_0,
        "ols_0_test": test_results_0,
        "ols_42_train": train_results_42,
        "ols_42_test": test_results_42,
    }

    for key, value in all_results.items():
        with open(data_path / key, "wb") as f:
            pickle.dump(value, f)

    ### Save some important plots to data/01_ols/plots.

    plots_path = data_path / "plots"
    plots_path.mkdir(parents=True, exist_ok=True)

    for key in all_results.keys():
        plot_and_save_diagnostics(all_results[key], plots_path / (key + ".png"))


if __name__ == "__main__":
    main()
