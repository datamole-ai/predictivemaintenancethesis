import numpy as np
from pmlib.modeling import create_artificial_labels


def series_scorer(score_func,
                  artificial_window: int = 1,
                  warning_time: int = 0,
                  evaluation_window: int = -1,
                  **kwargs):

    def trim_warning_time(data_series, warning_time: int):
        if warning_time < 1:
            return data_series
        return np.array([data[:-warning_time] for data in data_series])

    def trim_flatten(y_series):
        if evaluation_window > 0:
            y_series = [y[-evaluation_window:] for y in y_series]
        y_series = trim_warning_time(y_series, warning_time)
        y = np.hstack(y_series)
        return y

    def f(y_true_series, y_pred_series):
        y_true_series = [
            create_artificial_labels(y, artificial_window)
            for y in y_true_series
        ]
        return score_func(trim_flatten(y_true_series),
                          trim_flatten(y_pred_series),
                          **kwargs)

    return f
