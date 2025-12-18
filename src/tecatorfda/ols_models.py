from dataclasses import dataclass, field, fields

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import train_test_split


@dataclass
class ScikitLearnResults:
    X: np.ndarray
    y: np.ndarray
    model: BaseEstimator = field(metadata={"desc": "BaseEstimator"})
    split: str = field(metadata={"desc": "Train, test, or full data"})
    random_state: int | None = field(metadata={"desc": "Random state of model"})
    predicted: np.ndarray
    r2_score: float = field(metadata={"desc": "r2_score"})
    mse: float = field(metadata={"desc": "Mean squared error"})
    mae: float = field(metadata={"desc": "Mean absolute error"})
    mape: float = field(metadata={"desc": "Mean absolute percentage error"})
    leverage_scores: np.ndarray | None = None
    condition_number: float | None = field(
        default=None, metadata={"desc": "Condition number"}
    )


def print_attributes(obj: object) -> None:
    """Print the metadata of a dataclass above.

    Args:
        obj (object): The object.
    """
    for f in fields(obj):
        value = getattr(obj, f.name)
        desc = f.metadata.get("desc", "")
        if desc != "":
            print(f"{desc}: {value}")


def plot_and_save_diagnostics(results: ScikitLearnResults, filepath: str) -> None:
    """Plot actual vs. predicted values and residuals side-by-side.

    Args:
        results (ScikitLearnResults): The results of an OLS procedure.
        filepath (str): The path to which to save the plot.
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].plot(
        [min(results.y), max(results.y)],
        [min(results.y), max(results.y)],
        c="black",
        alpha=0.5,
    )
    ax[0].scatter(results.y, results.predicted, s=1)
    ax[0].set_title(rf"Train actual vs. predicted ($R^{2}={results.r2_score:.4f}$)")
    ax[0].set_xlabel("Actual")
    ax[0].set_ylabel("Predicted")

    ax[1].plot([min(results.y), max(results.y)], [0, 0], c="black", alpha=0.5)
    ax[1].scatter(results.y, results.y - results.predicted, s=1)
    ax[1].set_title(rf"Train residuals ($R^{2}={results.r2_score:.4f}$)")
    ax[1].set_xlabel("Predicted")
    ax[1].set_ylabel("Residual")

    fig.tight_layout()
    fig.savefig(filepath, dpi=450, bbox_inches="tight")
    plt.close(fig)


def holdout_split80(X: np.ndarray, y: np.ndarray, random_state: int | None = None):
    """We want to use a constant 80/20 train/test split throughout, so we fix that.

    Args:
        X (np.ndarray): Regressor matrix.
        y (np.ndarray): Response vector (shape (n, 1)).
        random_state (int | None, optional): random_state to use. Defaults to None.
    """
    return train_test_split(X, y, test_size=0.2, random_state=random_state)


def leverage_scores(X: np.ndarray) -> np.ndarray:
    """Given a data set, compute the leverage matrix, or the diagonal of the hat matrix.

    Args:
        X (np.ndarray): The predictor matrix.

    Returns:
        np.ndarray: The leverage values.
    """
    hat = X @ np.linalg.inv(X.T @ X) @ X.T

    return np.diag(hat)


def condition_number(X: np.ndarray) -> float:
    """Given a data set, compute the condition number, or measure of multicollinearity.

    Args:
        X (np.ndarray): The predictor matrix.

    Returns:
        float: The condition number, or ratio of largest to smallest eigenvalue.
    """
    singular_values = np.linalg.svd(X, compute_uv=False)

    return singular_values.max() / singular_values.min()


def perform_ols(X: np.ndarray, y: np.ndarray) -> ScikitLearnResults:
    """Perform OLS on a data set.

    Args:
        X (np.ndarray): Regressor matrix.
        y (np.ndarray): Response vector (shape (n, 1)).

    Returns:
        ScikitLearnResults: Results.
    """
    y.reshape(-1, 1)
    # Fit. Don't need to standardize for OLS.
    lr_model = LinearRegression()
    lr_model.fit(X, y)

    # Predict.
    predicted = lr_model.predict(X)

    # Define the results.
    results = ScikitLearnResults(
        X=X,
        y=y,
        model=lr_model,
        split="full",
        random_state=None,
        predicted=predicted,
        r2_score=r2_score(y, predicted),
        mse=mean_squared_error(y, predicted),
        mae=mean_absolute_error(y, predicted),
        mape=mean_absolute_percentage_error(y, predicted),
        leverage_scores=leverage_scores(X),
        condition_number=condition_number(X),
    )

    return results


def perform_ols_holdout(
    X: np.ndarray, y: np.ndarray, random_state: int | None = None
) -> tuple[ScikitLearnResults, ScikitLearnResults]:
    """Perform OLS on a train/test split.

    Args:
        X (np.ndarray): Regressor matrix.
        y (np.ndarray): Response vector (shape (n, 1)).
        random_state (int | None, optional): random_state to use. Defaults to None.

    Returns:
        tuple[ScikitLearnResults, ScikitLearnResults]: Train and test results.
    """
    y.reshape(-1, 1)
    # Split the data.
    X_train, X_test, y_train, y_test = holdout_split80(X, y, random_state)

    # Fit. Don't need to standardize for OLS.
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # Predict.
    predicted_train = lr_model.predict(X_train)
    predicted_test = lr_model.predict(X_test)

    # Define the training results.
    train_results = ScikitLearnResults(
        X=X_train,
        y=y_train,
        model=lr_model,
        split="train",
        random_state=random_state,
        predicted=predicted_train,
        r2_score=r2_score(y_train, predicted_train),
        mse=mean_squared_error(y_train, predicted_train),
        mae=mean_absolute_error(y_train, predicted_train),
        mape=mean_absolute_percentage_error(y_train, predicted_train),
        leverage_scores=leverage_scores(X_train),
        condition_number=condition_number(X_train),
    )

    # Define the testing results.
    test_results = ScikitLearnResults(
        X=X_test,
        y=y_test,
        model=lr_model,
        split="test",
        random_state=random_state,
        predicted=predicted_test,
        r2_score=r2_score(y_test, predicted_test),
        mse=mean_squared_error(y_test, predicted_test),
        mae=mean_absolute_error(y_test, predicted_test),
        mape=mean_absolute_percentage_error(y_test, predicted_test),
    )

    return train_results, test_results


def perform_ols_cv(
    X: np.ndarray, y: np.ndarray, random_state: int | None = None
) -> None:
    pass
