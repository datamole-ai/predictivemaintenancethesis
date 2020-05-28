from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from typing import Any, List, Union


_LOGGER = logging.getLogger(__name__)


def create_artificial_labels(labels: Union[np.array, pd.Series],
                             artificial_window: int) -> pd.Series:
    """
    Create a series of artificial labels by extending labels to the length
    of artificial window. The function takes as an input one time series.

    Parameters
    ----------
    labels : np.array or pd.Series
        original labels - one time series
    artificial_window : int
        length of a window to extend original labels to

    Returns
    -------
    pd.Series
        artificial labels
    """

    if artificial_window < 1:
        raise ValueError('Artificial window must be >= 1')

    if artificial_window == 1:
        return labels

    labels = np.array(labels, copy=True)

    if labels[-1]:
        labels[-artificial_window:] = 1

    return labels


def smooth_predictions(y_pred: np.array,
                       smoothing_window: int,
                       smoothing_threshold: int
                       ) -> np.array:
    """
    Smooth predictions.

    Parameters
    ----------
    y_pred : np.array or pd.Series
        Predictions which evaluate to True or False.
    smoothing_window : int
        Length of the window.
    smoothing_threshold : int
        Number of predictions in the window needed
        for an output prediction to be True.

    Returns
    -------
    pd.Series : Predictions of the same length as the input y_pred.
                A prediction is set to True if a number of positive
                predictions in last 'smoothing_window' days is greater or equal
                to 'smoothing_threshold'.
    """

    if smoothing_window < 2:
        return y_pred

    return (
        pd.Series(y_pred)
        .rolling(smoothing_window, min_periods=1)
        .mean()
        .ge(smoothing_threshold)
        .astype(int)
        .values
    )


class PredictiveMaintenanceModel(BaseEstimator):
    """
    A classifier using artificial labeling in fit method
    and smoothing of predictions in predict method

    Parameters
    ----------
    estimator : sklearn estimator
        An sklearn estimator to use for prediction.
    artificial_window : int
        The window length to use for artificial labeling before training.
    smoothing_window : int
        The window length to use for smoothing of the raw predictions.

    """

    def __init__(self, estimator: Any = DecisionTreeClassifier(),
                 artificial_window: int = 1, smoothing_window: int = 1,
                 decision_threshold: float = 0.5):
        self.estimator = estimator
        self.artificial_window = artificial_window
        self.smoothing_window = smoothing_window
        self.decision_threshold = decision_threshold

    def __create_artificial_labels(self, y: np.ndarray) -> np.ndarray:
        y_artificial = []
        for series_y in y:
            y_artificial.append(
                create_artificial_labels(series_y, self.artificial_window)
            )
        y_artificial = np.concatenate(y_artificial)
        return y_artificial

    def fit(self, X: np.ndarray, y: np.ndarray,
            **fit_params) -> PredictiveMaintenanceModel:
        """
        Fit the classifier.

        Parameters
        ----------
        X: np.ndarray
           3D array having series as first dimension,
           features as second dimension and time stamps as third dimension
        y: np.ndarray
           2D array having series as first dimension and time stamps
           as second dimension containg boolean flag whether
           a failure occured

        Returns
        -------
        PredictiveMaintenanceClassifier
            self
        """

        _LOGGER.info('Creating artificial labels...')
        y_artificial = self.__create_artificial_labels(y)

        _LOGGER.info('Flattening...')
        X_flattened = np.concatenate(X)

        _LOGGER.info(f'Fitting X.shape={X_flattened.shape},'
                     f' y.shape={y_artificial.shape}...')
        self.estimator.fit(X_flattened, y_artificial, **fit_params)

        _LOGGER.info('Model fitted')

        return self

    def predict(self,
                X_series: np.ndarray,
                decision_threshold: float = None,
                clip_lower_zero: bool = True
    ) -> np.ndarray:
        """
        Make prediction for X.

        Parameters
        ----------
        X : np.ndarray
            3D array having series as first dimension, features
            as a second dimension and time stamps as third dimension

        Returns
        -------
        np.ndarray
            Smoothed predictions for X - 2D array having series as first
            dimension and time stamps as second dimension.
        """
        y_pred_series = []
        for X in X_series:
            if callable(getattr(self.estimator, 'predict_proba', None)):
                y_score = self.estimator.predict_proba(X)[:, 1]
                y_pred = y_score >= self.decision_threshold
            else:
                y_pred = self.estimator.predict(X)
            if clip_lower_zero:
                y_pred = pd.Series(y_pred).clip(lower=0).values
            y_pred_series.append(y_pred)
        return y_pred_series

    def predict_proba(self, X_series: List[np.array]) -> List[np.array]:
        """
        Predict probabilities for X.

        Parameters
        ----------
        X : np.ndarray
            3D array having series as first dimension, features
            as a second dimension and time stamps as third dimension

        Returns
        -------
        np.ndarray
            Smoothed probabilities by moving window average of size
            'smoothing window'
        """
        y_score_series = []
        for X in X_series:
            y_score = self.estimator.predict_proba(X)[:, 1]
            y_score = (
                pd.Series(y_score)
                .rolling(self.smoothing_window, min_periods=1)
                .mean()
            )
            y_score_series.append(y_score)
        return y_score_series
