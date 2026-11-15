import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from vsensor.conformal import ConformalRegressor, coverage
from vsensor.data import add_time_features


def test_conformal_coverage_on_exchangeable_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 1))
    y = 3 * X[:, 0] + rng.normal(scale=1.0, size=4000)
    cr = ConformalRegressor(LinearRegression(), alpha=0.1).fit(X[:1000], y[:1000], X[1000:2000], y[1000:2000])
    _, lo, hi = cr.predict(X[2000:])
    cov = coverage(y[2000:], lo, hi)
    assert 0.87 <= cov <= 0.93          # ~90% when data are exchangeable


def test_time_features_use_only_the_past():
    df = pd.DataFrame({
        "date": pd.date_range("2015-02-02 08:00", periods=10, freq="min"),
        "Temperature": 20.0, "Humidity": 30.0, "HumidityRatio": 0.004,
        "Light": [0, 0, 500, 500, 500, 0, 0, 0, 0, 0], "CO2": 450, "Occupancy": 0,
    })
    f = add_time_features(df)
    # at row 2 the lights just turned on: only 1 "on" minute counted so far
    assert f.loc[2, "lights_on_30m"] == 1
    assert f.loc[4, "lights_on_30m"] == 3
    # changing a future value must not change past features
    df2 = df.copy(); df2.loc[9, "Light"] = 999
    assert add_time_features(df2).loc[:8, "lights_on_30m"].equals(f.loc[:8, "lights_on_30m"])
