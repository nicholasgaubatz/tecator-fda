import json

import numpy as np
from pathlib import Path
import pandas as pd
from skfda.datasets import fetch_tecator

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def fetch_tecator_fat() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Use scikit-fda to load the Tecator data set (215x100 and 215x1).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, np.ndarray]: 215x100 wavelength data set,
            215x1 fat data set, and shape (100,) wavelength grid data.
    """
    X, y = fetch_tecator(return_X_y=True, as_frame=True)
    grid = X.iloc[:, 0].values.grid_points[0]
    wavelengths = pd.DataFrame(X.iloc[:, 0].values.data_matrix.squeeze(), columns=grid)
    fat = pd.DataFrame(y["fat"].values, columns=["y"])

    return wavelengths, fat, grid


def generate_tecator_fat(location: str | None = None) -> None:
    """Use scikit-fda to load the Tecator data set.

    Args:
        location (str, optional): The directory to which the data are saved.
            If None, uses "data" in the repo root.
    """
    # If no directory is given, default to <repo_root>/data.
    if location is None:
        # Get the directory containing this file.
        src_dir = Path(__file__).parent
        # Assume repo root is parent of parent of src.
        repo_root = src_dir.parent.parent
        # Default data directory.
        location = repo_root / "data"
    else:
        location = Path(location)
        location = location.resolve()

    location.mkdir(parents=True, exist_ok=True)

    # Fetch the data.
    wavelength_df, fat_df, grid = fetch_tecator_fat()

    # Create a metadata dict for the grid.
    grid_metadata = {
        "dataset": "Tecator",
        "source": "https://lib.stat.cmu.edu/datasets/tecator",
        "wavelengths": grid.tolist(),
        "wavelength_unit": "nm",
        "notes": "100-point discretization per sample",
    }

    # Save the data.
    wavelength_df.to_csv(location / "tecator.csv", index=False)
    fat_df.to_csv(location / "fat.csv", index=False)
    with open(location / "metadata.json", "w") as f:
        json.dump(grid_metadata, f, indent=4)


def load_tecator_fat(
    location: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list, str]:
    """Load the Tecator data set from a pre-saved location.

    Args:
        location (str | None, optional): The directory in which the data are saved.
            If None, uses "data" in the repo root.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, list, str]: Tecator data, fat data,
            the wavelength grid, and the wavelength unit.
    """
    tecator_df = pd.read_csv("../data/tecator.csv")
    fat_df = pd.read_csv("../data/fat.csv")

    with open("../data/metadata.json", "r") as f:
        metadata = json.load(f)
    wavelength_grid = metadata["wavelengths"]
    wavelength_unit = metadata["wavelength_unit"]

    return tecator_df, fat_df, wavelength_grid, wavelength_unit
