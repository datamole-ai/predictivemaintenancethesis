import numpy as np
import pandas as pd


def prognostic_horizon(y_true, y_pred, alpha):

    def _within_boundaries(x, lower=-np.inf, upper=np.inf):
        return np.array((lower <= x) & (x <= upper))

    lower, upper = y_true - alpha, y_true + alpha
    ts = (
        pd.Series(np.r_[0, _within_boundaries(y_pred, lower, upper), 1])
        .astype(bool)
        .astype(int)
        .diff()
    )
    ph_index = ts[ts == 1].index[-1]
    ph_index -= 1  # Because of the np.r_[0,..] - the ts_diff array was longer
    return len(y_true) - ph_index


def mean_prognostic_horizon(y_true_series, y_pred_series, alpha):
    return np.mean([
        prognostic_horizon(y_true, y_pred, alpha)
        for y_true, y_pred in zip(y_true_series, y_pred_series)
    ])


def mean_asymmetrically_weighted_percentage_error(y_true, y_pred):

    def gamma(x):
        return -2 * x if x <= 0 else x

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    errs = (y_true - y_pred) / y_true
    return np.mean([gamma(err) for err in errs]) * 100


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
