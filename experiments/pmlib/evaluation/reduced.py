import numpy as np
import pandas as pd


class _FailurePredictionScorer:

    def __init__(self, y_true_series, y_pred_series, m_t, w_t):

        if w_t < 0:
            raise(ValueError("Warning time must be positive"))
        if m_t < 0:
            raise(ValueError("Monitoring time must be positive"))
        if m_t <= w_t:
            raise(ValueError('Monitoring time must be bigger than warning'
                             ' time'))

        real_true = []
        real_false = []
        total_target_events = 0

        # Calculate real_true, real_false and total_target_events
        for y_true, y_pred in zip(y_true_series, y_pred_series):

            y_true = np.array(y_true, copy=True)
            y_pred = np.array(y_pred, copy=True)

            # Check if EoL is at the end of series
            if y_true[-1] == 0:
                real_true.append(np.array([]))
                real_false.append(y_pred)
            else:
                total_target_events += 1
                _m_t = np.min([m_t, len(y_true)])
                if w_t > 0:
                    real_true.append(y_pred[-_m_t:-w_t])
                else:
                    real_true.append(y_pred[-_m_t:])
                real_false.append(y_pred[:-_m_t])

        self.real_true_series = real_true
        self.real_false_series = real_false
        self.m_t = m_t
        self.w_t = w_t
        self.total_target_events = total_target_events

    @property
    def tp(self):
        return np.hstack(self.real_true_series).sum()

    @property
    def fp(self):
        return np.hstack(self.real_false_series).sum()

    @property
    def tn(self):
        return (np.hstack(self.real_false_series) == 0).sum()

    @property
    def fn(self):
        return (np.hstack(self.real_true_series) == 0).sum()

    @property
    def discounted_fp(self):
        discounted_fp = 0
        for s in self.real_false_series:
            discounted_fp += (
                pd.Series(np.array(s, copy=True))
                .rolling(self.m_t, min_periods=1)
                .mean()
                .gt(0)
                .sum()
            ) / self.m_t
        return discounted_fp

    @property
    def target_events_predicted(self):
        return np.sum([
            np.r_[x, 0].max()  # add 0 in case x is an empty array
            for x in self.real_true_series
        ])
