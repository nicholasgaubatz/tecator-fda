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
        "n_samples": wavelength_df.shape[0],
        "n_grid": wavelength_df.shape[1],
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

    tecator_df = pd.read_csv(location / "tecator.csv")
    fat_df = pd.read_csv(location / "fat.csv")

    with open(location / "metadata.json", "r") as f:
        metadata = json.load(f)
    wavelength_grid = metadata["wavelengths"]
    wavelength_unit = metadata["wavelength_unit"]

    return tecator_df, fat_df, wavelength_grid, wavelength_unit


def generate_and_load_tecator_data_iid_noise(original_data_location, new_data_location, random_seed, alpha=0., row_idx=None, col_idx=None):
    location = original_data_location.resolve()
    new_data_location = new_data_location.resolve()

    new_data_location.mkdir(parents=True, exist_ok=True)

    tecator_df = pd.read_csv(location / "tecator.csv")
    fat_df = pd.read_csv(location / "fat.csv")

    # If requested, take the subset of rows.
    if row_idx is not None:
        tecator_df = tecator_df.iloc[row_idx]
        fat_df = fat_df.iloc[row_idx]

    # Open up metadata.
    with open(location / "metadata.json", "r") as f:
        metadata = json.load(f)
    wavelength_grid = metadata["wavelengths"]
    wavelength_unit = metadata["wavelength_unit"]

    # If requested, take the subset of columns.
    if col_idx is not None:
        # allow either positional indices/mask/slice or explicit column names
        if isinstance(col_idx, (slice, list, tuple, np.ndarray)) and not (
            isinstance(col_idx, (list, tuple, np.ndarray)) and len(col_idx) > 0 and isinstance(col_idx[0], str)
        ):
            tecator_df = tecator_df.iloc[:, col_idx]
        else:
            tecator_df = tecator_df.loc[:, col_idx]

        # if they subset by names, map names -> positions; if by positions, use directly
        if not (isinstance(col_idx, (slice, list, tuple, np.ndarray)) and not (
            isinstance(col_idx, (list, tuple, np.ndarray)) and len(col_idx) > 0 and isinstance(col_idx[0], str)
        )):
            # column names case
            col_pos = [tecator_df.columns.get_loc(c) for c in tecator_df.columns]
        else:
            # positional case already applied; positions are 0..n_selected-1 now
            col_pos = np.arange(tecator_df.shape[1])

        wavelength_grid = np.asarray(wavelength_grid)[col_pos].tolist()

    # Add the noise, if requested.
    if alpha != 0.:
        noisy_tecator_df = add_gaussian_noise_per_column(df=tecator_df, alpha=alpha, random_seed=random_seed)
    else:
        noisy_tecator_df = tecator_df.copy()

    # Create a metadata dict for the grid.
    grid_metadata = {
        "dataset": f"Tecator with noise level {alpha}",
        "source": "https://lib.stat.cmu.edu/datasets/tecator",
        "n_samples": noisy_tecator_df.shape[0],
        "n_grid": noisy_tecator_df.shape[1],
        "wavelengths": list(wavelength_grid),
        "wavelength_unit": "nm",
        "notes": f"{len(list(wavelength_grid))}-point discretization per sample; "
                 f"row_idx={row_idx is not None}; col_idx={col_idx is not None}",
    }

    # Reset indices to keep things clean.
    noisy_tecator_df = noisy_tecator_df.reset_index(drop=True)
    fat_df = fat_df.reset_index(drop=True)

    # Save the data.
    noisy_tecator_df.to_csv(new_data_location / "tecator.csv", index=False)
    fat_df.to_csv(new_data_location / "fat.csv", index=False)
    with open(new_data_location / "metadata.json", "w") as f:
        json.dump(grid_metadata, f, indent=4)

    return noisy_tecator_df, fat_df, wavelength_grid, wavelength_unit


def add_gaussian_noise_per_column(df, alpha, random_seed):
    """Add noise to functional data based on columnwise std. Alpha controls the level of noise.

    Args:
        X (_type_): _description_
        alpha (_type_): _description_
        random_seed (_type_): _description_
    """
    X = df.to_numpy(dtype=float)
    rng = np.random.default_rng(random_seed)
    sd = X.std(axis=0, ddof=1) # Shape (n_grid,)
    noise = rng.normal(0.0, alpha * sd, size=X.shape)

    result = df.copy()
    result.loc[:, :] = X + noise

    return result
