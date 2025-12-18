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
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ScikitLearnResults:
    X: np.ndarray
    y: np.ndarray
    model: BaseEstimator = field(metadata={"desc": "BaseEstimator"})
    split: str = field(metadata={"desc": "Train, test, or full data"})
    random_state: int | None = field(metadata={"desc": "Random state of model"})
    predicted: np.ndarray
    residuals: np.ndarray
    r2_score: float = field(metadata={"desc": "r2_score"})
    mse: float = field(metadata={"desc": "Mean squared error"})
    mae: float = field(metadata={"desc": "Mean absolute error"})
    mape: float = field(metadata={"desc": "Mean absolute percentage error"})
    leverage_scores: np.ndarray | None = None
    condition_number: float | None = field(
        default=None, metadata={"desc": "Condition number"}
    )
    cooks_distances: np.ndarray | None = None


@dataclass
class ScikitLearnCVResults:
    X: np.ndarray
    y: np.ndarray
    model: BaseEstimator = field(metadata={"desc": "BaseEstimator"})
    random_state: int | None = field(metadata={"desc": "Random state of model"})
    r2_scores: np.ndarray
    r2_mean: float = field(metadata={"desc": "Mean R^2 of CV fits"})
    r2_std: float = field(metadata={"desc": "Standard deviation of CV fits"})
    mse: np.ndarray
    mae: np.ndarray
    mape: np.ndarray


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
    ax[0].set_title(
        rf"{results.split.capitalize()} actual vs. predicted, state={results.random_state} ($R^{2}={results.r2_score:.4f}$)"
    )
    ax[0].set_xlabel("Actual")
    ax[0].set_ylabel("Predicted")

    ax[1].plot([min(results.y), max(results.y)], [0, 0], c="black", alpha=0.5)
    ax[1].scatter(results.y, results.y - results.predicted, s=1)
    ax[1].set_title(
        rf"{results.split.capitalize()} residuals, state={results.random_state}"
    )
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
    """Given a data set X, compute the condition number of X'X for multicollinearity.

    Args:
        X (np.ndarray): The predictor matrix.

    Returns:
        float: The condition number, or ratio of largest to smallest eigenvalue.
    """
    eigenvalues = np.linalg.eig(X.T @ X).eigenvalues

    return eigenvalues.max() / eigenvalues.min()


def cooks_distances():
    # TODO
    pass


def perform_ols(X: np.ndarray, y: np.ndarray) -> ScikitLearnResults:
    """Perform OLS on a data set.

    Args:
        X (np.ndarray): Regressor matrix.
        y (np.ndarray): Response vector (shape (n, 1)).

    Returns:
        ScikitLearnResults: Results.
    """
    y.reshape(-1, 1)
    # Fit. Don't need to standardize for OLS, but we do for consistency later.
    lr_model = make_pipeline(StandardScaler(), LinearRegression())
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
        residuals=y - predicted,
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

    # Fit. Don't need to standardize for OLS, but we do for consistency later.
    lr_model = make_pipeline(StandardScaler(), LinearRegression())
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
        residuals=y_train - predicted_train,
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
        residuals=y_test - predicted_test,
        r2_score=r2_score(y_test, predicted_test),
        mse=mean_squared_error(y_test, predicted_test),
        mae=mean_absolute_error(y_test, predicted_test),
        mape=mean_absolute_percentage_error(y_test, predicted_test),
    )

    return train_results, test_results


def perform_ols_cv(
    X: np.ndarray, y: np.ndarray, random_state: int | None = None
) -> None:
    y.reshape(-1, 1)
    # Perform 10-fold cross-validation 10 times.
    lr_model = make_pipeline(StandardScaler(), LinearRegression())
    cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=random_state)

    # Evaluate all the metrics.
    r2_scores = cross_val_score(lr_model, X, y, cv=cv, scoring="r2")
    mse_scores = cross_val_score(
        lr_model, X, y, cv=cv, scoring="neg_mean_squared_error"
    )
    mae_scores = cross_val_score(
        lr_model, X, y, cv=cv, scoring="neg_mean_absolute_error"
    )
    mape_scores = cross_val_score(
        lr_model, X, y, cv=cv, scoring="neg_mean_absolute_percentage_error"
    )

    # Define the results.
    results = ScikitLearnCVResults(
        X=X,
        y=y,
        model=lr_model,
        random_state=random_state,
        r2_scores=r2_scores,
        r2_mean=r2_scores.mean(),
        r2_std=r2_scores.std(),
        mse=mse_scores,
        mae=mae_scores,
        mape=mape_scores,
    )

    return results
