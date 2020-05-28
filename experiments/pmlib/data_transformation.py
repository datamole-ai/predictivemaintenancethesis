import pandas as pd

from typing import List


def split_data(df: pd.DataFrame, feature_cols: List[str], target_col: str,
               series_id_col: str, subject_id_col: str) -> pd.DataFrame:
    """

    Splits dataset into X (features), y (target - events)
    and groups (identifiers based on `subject_id_col` for cross validation).

    All the X, y and groups are 3D array where first dimension stands
    for each series id, second dimension stands for features
    and third dimension stands for time stamp.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset.
    feature_cols: list
        List of column names containing features.
    target_col: str
        Name of the column containing events - values to be predicted.
    series_id_col : str
        Name of the column containing series_id.
    subject_id_col : str
        Name of the column containing identification of subjects.
        Used for creating groups and for creating series_id
        if series_id_col not supplied.

    Returns
    -------
    np.aarray
        X
    np.array
        y
    np.array
        groups
    """

    grouped = df.groupby(series_id_col)

    X = (
        grouped
        .apply(lambda x: x[feature_cols].values)
        .values
    )

    y = (
        grouped
        .apply(lambda x: x[target_col].astype(int).values)
        .values
    )

    groups = grouped[subject_id_col].max().values

    return X, y, groups
