from typing import List

import numpy as np
from pmlib.evaluation.range_based import _TSMetric
from pmlib.evaluation.utils import series_scorer
from pmlib.evaluation.reduced import _FailurePredictionScorer


def recall_reduced_score(
    y_true_series: List[List[int]],
    y_pred_series: List[List[int]],
    monitoring_time: int,
    warning_time: int,
    alpha: float,
) -> float:
    """Calculate reduced threshold.

    Parameters
    ----------
    y_true_series : List[List[Int]]
        Array of series with actual labels
    y_pred_series : List[List[Int]]
        Array of series with predictions
    monitoring_time : int
        The length of time window for which we want to predict whether
        a failure will occur.
    warning_time : int
        The minimal number of time steps prior to the failure in order
        for the prediction to be considered useful.
        Warning time

    Returns
    -------
    float
        Recall score
    """
    def calc_tp(y_true, y_pred, alpha):
        metric = _TSMetric(metric_option="time-series",
                           alpha_r=alpha,
                           cardinality='one',
                           bias_r='flat')
        return metric.tp(y_true, y_pred)

    scorer_tp = series_scorer(
        calc_tp, monitoring_time, warning_time, alpha=alpha)
    tp = scorer_tp(y_true_series, y_pred_series)
    p = np.sum(np.hstack(y_true_series))

    rec = tp / p
    return rec


def precision_reduced_score(
    y_true_series: List[List[int]],
    y_pred_series: List[List[int]],
    monitoring_time: int,
    warning_time: int,
    alpha: float,
) -> float:
    """Calculate reduced precision.

    Parameters
    ----------
    y_true_series : List[List[Int]]
        Array of series with actual labels
    y_pred_series : List[List[Int]]
        Array of series with predictions
    monitoring_time : int
        The length of time window for which we want to predict whether
        a failure will occur.
    warning_time : int
        The minimal number of time steps prior to the failure in order
        for the prediction to be considered useful.

    Returns
    -------
    float
        Reduced precision score
    """
    scorer = _FailurePredictionScorer(
        y_true_series, y_pred_series, monitoring_time, warning_time)

    def calc_tp(y_true, y_pred, alpha):
        metric = _TSMetric(metric_option="time-series",
                           alpha_r=alpha,
                           cardinality='one',
                           bias_r='flat')
        return metric.tp(y_true, y_pred)

    scorer_tp = series_scorer(
        calc_tp, monitoring_time, warning_time, alpha=alpha)

    tp = scorer_tp(y_true_series, y_pred_series)
    fp = scorer.discounted_fp
    prec = tp / (tp + fp)

    return prec


def f1_reduced_score(
    y_true_series: List[List[int]],
    y_pred_series: List[List[int]],
    monitoring_time: int,
    warning_time: int,
    alpha: float,
) -> float:
    scorer = _FailurePredictionScorer(
        y_true_series, y_pred_series, monitoring_time, warning_time)

    def calc_tp(y_true, y_pred, alpha):
        metric = _TSMetric(metric_option="time-series",
                           alpha_r=alpha,
                           cardinality='one',
                           bias_r='flat')
        return metric.tp(y_true, y_pred)

    scorer_tp = series_scorer(
        calc_tp, monitoring_time, warning_time, alpha=alpha)
    tp = scorer_tp(y_true_series, y_pred_series)

    fp = scorer.discounted_fp
    p = scorer.total_target_events

    prec = tp / (tp + fp)
    rec = tp / p
    return np.nan_to_num((2*prec*rec) / (prec + rec))
