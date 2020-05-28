from typing import Any

import pandas as pd


def create_series_id(data: pd.DataFrame, event_col: str,
                     subject_id_col: str = None, time_order_col: str = None,
                     max_gap: Any = None) -> pd.DataFrame:
    """
    Creates pandas.Series of the same length as input data frame 'data'
    containing series id for each row. The series id is created as follows

        #. The first row has series id equal to 1.
        #. For every row, if any of the conditions below are true series id
           is inremented by 1.
            - The **previous row's** value in column `event_col` is greater
              than 0 (an event happened on previous row and this row starts
              new series).
            - This row's value of column `subject_id_col` is different
              from previous row. (only if `subject_id_col` is supplied)
            - Difference of values of column `time_order_col` of this row
              and the previous row is greater than `max_gap`.
              (only if `time_order_col` and `max_gap` is supplied)

    Parameters
    ----------
    data : pd.DataFrame
        Data containing `sort_by_col`, `event_cols` and `subject_id_col`.
    event_col : str
        List of columns that contain binary labels of failures.
    subject_id_col : List[str]
        Columns which identify different time series.
        E.g. ['DeviceId'] or ['CustomerNumber'].
        These columns are used for pd.DataFrame.groupby operation.
    time_order_col : str
        Name of the column which gives order to a time series of each subject.
    max_gap : pd.Timedelta
        Maximal difference betweeen two values of `time_order_col`.
        If the difference is higher, the series id is incremented by 1.

    Returns
    -------
    pd.Series
        Series ids.

    Examples
    --------
    >>> df['SeriesId'] = create_series_id(
    >>>    data=df,
    >>>    event_cols='Event',
    >>>    subject_id_col='SubjectId')
    >>>
    >>> gap = pd.TimeDelta(1, 'D')
    >>> df['SeriesId_Gap=1day'] = create_series_id(
    >>>    data=df,
    >>>    time_order_col='Date',
    >>>    event_cols='Event',
    >>>    subject_id_col='SubjectId',
    >>>    max_gap=gap)
    >>> display(df)
        Date	    Event   SubjectId	SeriesId_Gap=None	SeriesId_Gap=1day
    0	2019-01-01	0	    A	        1                   1
    1	2019-01-02	1	    A	        1                   1
    2	2019-01-03	0	    A	        2                   2
    3	2019-01-05	1	    A	        2                   3
    4	2019-01-06	0	    A	        3                   4
    5	2019-01-07	0	    A	        3                   4
    6	2019-01-01	0	    B	        4                   5
    7	2019-01-02	0	    B	        4                   5
    8	2019-01-03	0	    B	        4                   5
    9	2019-01-04	1	    C	        5                   6
    10	2019-01-05	1	    C	        6                   7
    11	2019-01-06	1	    C	        7                   8
    """

    series_id = pd.Series(0, index=data.index)

    # Events
    event_occurred = (
        data[event_col]
        .shift()  # previous day
        .gt(0)  # True if any event occured
    )
    series_id[event_occurred] = 1

    # Date Gaps
    if max_gap is not None and time_order_col is not None:
        date_diff = data[time_order_col].diff()
        significant_date_gap = date_diff.gt(max_gap)
        series_id[significant_date_gap] = 1

    # Subject changes
    if subject_id_col is not None:
        id_changed = data[subject_id_col] != data[subject_id_col].shift()
        series_id[id_changed] = 1

    series_id = series_id.cumsum()
    return series_id


def filter_short_series(data: pd.DataFrame,
                        min_series_len: int) -> pd.DataFrame:
    """Filters out series shorter than `min_series_len`.
    DataFrame `data` must contain a column `SeriesId`.

    Parameters
    ----------
    data : pd.DataFrame
        Data containing column SeriesId
    min_series_len : int
        Minimal length that each group of SeriesId must have.

    Returns
    -------
    pd.DataFrame
        Data without rows containing SeriesId that had less than
        `min_series_len` occurences.
    """
    short_series = (
        data
        .groupby('SeriesId')
        .size()
        .loc[lambda x: x.lt(min_series_len)]
        .index
    )
    filtered_data = data[~data['SeriesId'].isin(short_series)].copy()
    return filtered_data
