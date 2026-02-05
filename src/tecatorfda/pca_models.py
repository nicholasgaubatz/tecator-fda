import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.extmath import svd_flip

class CustomPCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None, svd_flip_sign=True):
        """A scikit-learn-compatible transformer that we can use with repeated CV.

        Args:
            n_components (_type_, optional): _description_. Defaults to None.
            svd_flip_sign (bool, optional): _description_. Defaults to True.
        """
        self.n_components = n_components
        self.svd_flip_sign = svd_flip_sign

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        n, p = X.shape
        self.mean_ = X.mean(axis=0)

        Xc = X - self.mean_
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False) # full_matrices=False => U isn't square

        if self.svd_flip_sign:
            U, Vt = svd_flip(U, Vt, u_based_decision=True)

        self.singular_values_ = s
        self.components_ = Vt  # rows are components (like sklearn)

        self.explained_variance_ = (s**2) / (n - 1)

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        Xc = X - self.mean_
        Vt = self.components_
        k = self.n_components if self.n_components is not None else Vt.shape[0]
        return Xc @ Vt[:k].T
