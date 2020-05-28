import numpy as np
import pandas as pd

from typing import List


def unstack(df, col, groupby_cols, prefix=None):
    return (
        pd.concat(
            [
                df[groupby_cols],
                pd.get_dummies(df[col], prefix=prefix)
            ],
            axis=1
        )
        .groupby(groupby_cols, as_index=False)
        .sum()
    )


def create_features_last_event_time(
    data: pd.DataFrame,
    event_cols: List[str],
    time_col: str,
    subject_id_col: str = None,
    feature_name_format: str = 'last_time_%COL',
) -> pd.DataFrame:
    out_df = data[[subject_id_col, time_col]].copy()
    for event_col in event_cols:
        feature_name = feature_name_format.replace('%COL', event_col)
        out_df[feature_name] = np.datetime64('nat')
        out_df.loc[data[event_col].eq(1), feature_name] = data[time_col]
        out_df[feature_name] = (
            out_df
            .groupby(subject_id_col)
            [feature_name]
            .ffill()
        )
    return out_df


def prepare_events_data(events_data, base_data, subject_id_col, time_col):

    event_cols = events_data.drop(columns=[subject_id_col, time_col]).columns

    # IMPORTANT: Make sure its sorted
    df = events_data.sort_values([subject_id_col, time_col])

    # Add new features
    last_event_time_features = create_features_last_event_time(
        data=df,
        event_cols=event_cols,
        subject_id_col=subject_id_col,
        time_col=time_col,
        feature_name_format='time_since_%COL',
    )
    df = df.merge(last_event_time_features,
                  on=[subject_id_col, time_col],
                  how='inner')
    new_cols = df.drop(columns=[subject_id_col, time_col, *event_cols]).columns

    # Outer merge to have all the dates
    # (including dates prior to base data)
    df = (
        df
        .merge(base_data, on=[subject_id_col, time_col], how='outer')
        .sort_values([subject_id_col, time_col])
    )

    df[event_cols] = df[event_cols].fillna(0).astype(int)
    df[new_cols] = df[new_cols].shift(1)

    for col in new_cols:
        subject_first_nan_row_indices = (
            df
            .groupby(subject_id_col)
            [col]
            .head(1)
            .loc[lambda df: df.isna()]
            .index
        )
        df.loc[subject_first_nan_row_indices, col] = df[time_col]
    df[new_cols] = df.groupby(subject_id_col)[new_cols].ffill()

    df = (
        df
        # Right merge to have the right shape
        .merge(base_data, on=[subject_id_col, time_col], how='right')
        .sort_values([subject_id_col, time_col])
        # Reset changes made by sorting
        .reset_index(drop=True)
    )
    for col in new_cols:
        df[col] = (
            (df[time_col] - df[col])
            # .dt.total_seconds()
            # .shift()
            # .fillna(0)
            # .astype(int)
        )

    return df
