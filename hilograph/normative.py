"""Normative deviation (Sec. 3.1, "Normative deviation").

After training, the encoder is frozen. For every latent dimension k we fit age-
and sex-conditioned baselines mu_k(a, s) and sigma_k(a, s) on the healthy
training latent space with robust (Huber) quadratic regression, and score any
subject by the deviation

    d_k = (z_k - mu_k(a, s)) / sigma_k(a, s).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.linear_model import HuberRegressor

# 1 / Phi^{-1}(3/4): turns a median absolute deviation into a Gaussian std.
_MAD_TO_STD = 1.4826


class NormativeModel:
    def __init__(self, use_sex: bool = True, max_iter: int = 1000):
        self.use_sex = use_sex
        self.max_iter = max_iter

    def _design(self, age: np.ndarray, sex: Optional[np.ndarray]) -> np.ndarray:
        a = (np.asarray(age, dtype=float) - self.age_mean_) / self.age_std_
        cols = [a, a ** 2]
        if self.use_sex:
            if sex is None or np.isnan(np.asarray(sex, dtype=float)).any():
                raise ValueError("Sex is required for every subject when use_sex=True")
            cols.append(np.asarray(sex, dtype=float))
        return np.column_stack(cols)

    def fit(self, z: np.ndarray, age: np.ndarray, sex: Optional[np.ndarray] = None) -> "NormativeModel":
        self.age_mean_ = float(np.mean(age))
        self.age_std_ = float(np.std(age)) or 1.0
        x = self._design(age, sex)
        self.mean_models_, self.scale_models_, self.scale_calib_ = [], [], []
        for k in range(z.shape[1]):
            mean_k = HuberRegressor(max_iter=self.max_iter).fit(x, z[:, k])
            res = z[:, k] - mean_k.predict(x)
            # heteroscedastic scale: robust quadratic fit of log|residual| ...
            scale_k = HuberRegressor(max_iter=self.max_iter).fit(x, np.log(np.abs(res) + 1e-8))
            # ... calibrated so the standardised training residuals have unit robust std
            u = res / np.exp(scale_k.predict(x))
            self.mean_models_.append(mean_k)
            self.scale_models_.append(scale_k)
            self.scale_calib_.append(_MAD_TO_STD * np.median(np.abs(u - np.median(u))))
        return self

    def predict(self, age: np.ndarray, sex: Optional[np.ndarray] = None):
        """Normative mean and standard deviation of every latent dimension."""
        x = self._design(age, sex)
        mu = np.column_stack([m.predict(x) for m in self.mean_models_])
        sigma = np.column_stack([c * np.exp(m.predict(x))
                                 for m, c in zip(self.scale_models_, self.scale_calib_)])
        return mu, sigma

    def deviation(self, z: np.ndarray, age: np.ndarray, sex: Optional[np.ndarray] = None) -> np.ndarray:
        mu, sigma = self.predict(age, sex)
        return (z - mu) / np.maximum(sigma, 1e-8)
