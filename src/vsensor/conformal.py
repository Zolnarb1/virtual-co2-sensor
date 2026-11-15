"""Split conformal prediction: turn any regressor into one with calibrated intervals.

Idea (Vovk et al.; Lei et al., 2018):
  1. fit the model on a training part,
  2. on a held-out calibration part, compute absolute errors |y - y_hat|,
  3. the interval for a new point is y_hat ± q, where q is the (1 - alpha) quantile
     of the calibration errors (with a small finite-sample correction).
If calibration and test data are exchangeable, the interval contains the truth at
least (1 - alpha) of the time. With time series that assumption can break, which is
exactly what we measure.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class ConformalRegressor:
    model: object
    alpha: float = 0.1
    q: float | None = None

    def fit(self, X_fit, y_fit, X_cal, y_cal):
        self.model.fit(X_fit, y_fit)
        residuals = np.abs(np.asarray(y_cal) - self.model.predict(X_cal))
        n = len(residuals)
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.q = float(np.quantile(residuals, level, method="higher"))
        return self

    def predict(self, X):
        y_hat = self.model.predict(X)
        return y_hat, y_hat - self.q, y_hat + self.q


def coverage(y, lower, upper) -> float:
    y = np.asarray(y)
    return float(np.mean((y >= lower) & (y <= upper)))
