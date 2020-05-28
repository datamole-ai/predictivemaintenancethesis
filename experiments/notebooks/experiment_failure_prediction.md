---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.0
  kernelspec:
    display_name: pmlib
    language: python
    name: pmlib
---

<!-- #region Collapsed="false" -->
# Failure Prediction on Azure Telemetry Data Set
<!-- #endregion -->

```python Collapsed="false"
# Enable autoreloading
%reload_ext autoreload
%autoreload 2
```

```python Collapsed="false"
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import pandas as pd
import seaborn as sns
import string

from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (f1_score,
                             make_scorer,
                             recall_score,
                             precision_recall_curve,
                             precision_score)
from sklearn.model_selection import (GroupKFold,
                                     RandomizedSearchCV,
                                     train_test_split)
from tqdm import tqdm
from urllib.parse import urljoin
from urllib.request import urlopen
from xgboost import XGBClassifier

from pmlib.data_transformation import split_data
from pmlib.evaluation.classical import auprg_score
from pmlib.evaluation.event_based import (f1_reduced_score,
                                          precision_reduced_score,
                                          recall_reduced_score)
from pmlib.evaluation.utils import series_scorer
from pmlib.feature_extraction import (create_features_last_event_time,
                                      prepare_events_data, unstack)
from pmlib.modeling import PredictiveMaintenanceModel
from pmlib.preprocessing import create_series_id, filter_short_series
```

```python Collapsed="false"
N_JOBS = 32
RANDOM_STATE = 7
```

```python Collapsed="false"
!mkdir -p ./data/azure/
!mkdir -p ./images/azure/
```

<!-- #region Collapsed="false" -->
# Load Data
<!-- #endregion -->

```python Collapsed="false"
BASE_URL = 'https://azuremlsampleexperiments.blob.core.windows.net/datasets/'
DATA_DIR = Path('./data/azure')
```

```python Collapsed="false"
def load_azure_pdm_data_set():

    def _download_azure_data_set_file(file_name):
        try:
            df = pd.read_csv(DATA_DIR / file_name)
        except:
            with urlopen(urljoin(BASE_URL, file_name)) as f:
                df = pd.read_csv(f)
            df.to_csv(DATA_DIR / file_name, index=False)
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        return df
    
    telemetry = _download_azure_data_set_file('PdM_telemetry.csv')
    errors = _download_azure_data_set_file('PdM_errors.csv')
    machines = _download_azure_data_set_file('PdM_machines.csv')
    failures = _download_azure_data_set_file('PdM_failures.csv')
    maintenance = _download_azure_data_set_file('PdM_maint.csv')
    return telemetry, errors, machines, failures, maintenance
```

```python Collapsed="false"
telemetry, errors, machines, failures, maintenance = load_azure_pdm_data_set()
```

<!-- #region Collapsed="false" -->
# Preprocess Data
<!-- #endregion -->

```python Collapsed="false"
subject_id_col = 'machineID'
time_col = 'datetime'
```

<!-- #region Collapsed="false" -->
## Preprocess and Merge Data Sources
<!-- #endregion -->

<!-- #region Collapsed="false" -->
### Telemetry Data
<!-- #endregion -->

```python Collapsed="false"
telemetry_preprocessed = telemetry.set_index([subject_id_col, time_col]).add_prefix('telemetry_raw_').reset_index()
feature_cols_telemetry = telemetry.columns[telemetry.columns.str.startswith('telemetry_raw')]
```

```python Collapsed="false"
base_df = (
    telemetry[[subject_id_col, time_col]]
    .sort_values([subject_id_col, time_col])
    .reset_index(drop=True)
)

base_df.sample(5, random_state=RANDOM_STATE)
```

<!-- #region Collapsed="false" -->
### Machines Static Data
<!-- #endregion -->

```python Collapsed="false"
machines.sample(5, random_state=RANDOM_STATE)
```

```python Collapsed="false"
machines_preprocessed = (
    unstack(machines, col='model', groupby_cols='machineID')
    .merge(machines[['machineID', 'age']], on='machineID')
)
machines_preprocessed.sample(5, random_state=RANDOM_STATE)
```

<!-- #region Collapsed="false" -->
### Error Log
<!-- #endregion -->

```python Collapsed="false"
errors.sample(5, random_state=RANDOM_STATE)
```

```python Collapsed="false"
error_cols = [f'error{i}' for i in range(1, 6)]

errors_preprocessed = (
    errors
    .rename(columns={'errorID': 'error'})
    .pipe(unstack, col='error', groupby_cols=[subject_id_col, time_col], prefix=None)
    .merge(base_df, on=[subject_id_col, time_col], how='right')
    .fillna(0)
    .sort_values([subject_id_col, time_col])
    .reset_index(drop=True)
    .pipe(prepare_events_data, base_data=base_df, subject_id_col=subject_id_col, time_col=time_col)
)

feature_cols_errors = errors_preprocessed.columns[errors_preprocessed.columns.str.startswith('time_since')]
errors_preprocessed.sample(5, random_state=RANDOM_STATE)
```

<!-- #region Collapsed="false" -->
### Failures Log
<!-- #endregion -->

```python Collapsed="false"
failures.sample(5, random_state=RANDOM_STATE)
```

```python Collapsed="false"
failure_cols = [f'failure_comp{i}' for i in range(1, 5)]

failures_preprocessed = (
    failures
    .pipe(unstack, col='failure', groupby_cols=[subject_id_col, time_col], prefix='failure')
    .pipe(prepare_events_data, base_data=base_df, subject_id_col=subject_id_col, time_col=time_col)
)
feature_cols_failure = failures_preprocessed.columns[failures_preprocessed.columns.str.startswith('seconds_since')]
failures_preprocessed.sample(5, random_state=RANDOM_STATE)
```

<!-- #region Collapsed="false" -->
### Maintenance Logs
<!-- #endregion -->

```python Collapsed="false"
maintenance.sample(5, random_state=RANDOM_STATE)
```

```python Collapsed="false"
maintenance_cols = [f'maintenance_comp{i}' for i in range(1, 5)]

maintenance_preprocessed = (
    maintenance
    .sort_values([subject_id_col, time_col])
    .pipe(unstack, col='comp', groupby_cols=[subject_id_col, time_col], prefix='maintenance')
    .pipe(prepare_events_data, base_data=base_df, subject_id_col=subject_id_col, time_col=time_col)
)
feature_cols_maintenance = maintenance_preprocessed.columns[maintenance_preprocessed.columns.str.startswith('seconds_since')]
maintenance_preprocessed.sample(5, random_state=RANDOM_STATE)
```

<!-- #region Collapsed="false" -->
### Merge Data
<!-- #endregion -->

```python Collapsed="false"
merge_props = {
    'on': [subject_id_col, time_col],
    'how': 'left',
}

data = telemetry_preprocessed
for data_to_merge in [
    failures_preprocessed,
    maintenance_preprocessed,
    errors_preprocessed,
]:
    data = data.merge(data_to_merge)
data.sample(5, random_state=RANDOM_STATE)
```

```python Collapsed="false"
subject_id_col = 'machineID'
time_col = 'datetime'
failure_cols = data.columns[data.columns.str.startswith('failure_')]
maintenance_cols = data.columns[data.columns.str.startswith('maintenance_')]
error_cols = data.columns[data.columns.str.startswith('error_')]
time_since_cols = data.columns[data.columns.str.startswith('time_since_')]
telemetry_cols = data.columns[data.columns.str.startswith('telemetry_raw_')]

data[time_since_cols] = data[time_since_cols].apply(lambda x: x.dt.total_seconds())
```

<!-- #region Collapsed="false" -->
### Add Event Column, Filter Short Series and Add Series Id
<!-- #endregion -->

```python Collapsed="false"
df = data.copy()

df['Event'] = df[failure_cols].max(axis=1).eq(1).astype(int)
df['SeriesId'] = create_series_id(
    df, event_col='Event', subject_id_col=subject_id_col, time_order_col=time_col)
for i in np.arange(1, 5):
    df[f'Event_comp{i}'] = df[[f'failure_comp{i}']].max(axis=1).eq(1).astype(int)
    df[f'SeriesId_comp{i}'] = create_series_id(
        df, event_col=f'Event_comp{i}', subject_id_col=subject_id_col, time_order_col=time_col)
df = filter_short_series(df, min_series_len=60)

data_with_event = df
```

<!-- #region Collapsed="false" -->
## Extract Features
<!-- #endregion -->

<!-- #region Collapsed="false" -->
### Rolling Features of Telemetry Data
<!-- #endregion -->

```python Collapsed="false"
class FeatureExtractor:
    def __init__(self, column_name, agg_func, agg_name):
        self.column_name = column_name
        self.agg_func = agg_func
        self.agg_name = agg_name
        

agg_mapping = [
    FeatureExtractor(col, func, agg_name)    
    for col in telemetry_cols
    for func, agg_name in [
        (np.mean, f'{col}_mean'),
        (np.sum, f'{col}_sum'),
        (np.var, f'{col}_var'),
    ]    
]
```

```python Collapsed="false"
feature_cols = [
    *telemetry_cols,
    *time_since_cols,
]
```

```python Collapsed="false"
def extract_telemetry_features(df):
    df = df[1]
    for agg_mapper in agg_mapping:
        df[agg_mapper.agg_name] = (
            df
            [agg_mapper.column_name]
            .rolling(24*7, min_periods=1)
            .apply(agg_mapper.agg_func, raw=True)
        )
    return df
```

```python Collapsed="false"
feature_cols_telemetry = [x.agg_name for x in agg_mapping]
```

```python Collapsed="false"
%%time

pool = multiprocessing.Pool(N_JOBS)

dfs = pool.map(extract_telemetry_features, list(data_with_event.groupby('SeriesId')))
pool.close()
data_feature_extracted = pd.concat(dfs)
```

```python Collapsed="false"
feature_cols = set(feature_cols)
feature_cols.update(set(feature_cols_telemetry))
feature_cols = list(feature_cols)
sorted(feature_cols)
```

```python Collapsed="false"
data_feature_extracted.sample(5, random_state=RANDOM_STATE)
```

<!-- #region Collapsed="false" -->
## Data Splitting
<!-- #endregion -->

```python Collapsed="false"
subjects_train, subjects_test = train_test_split(data[subject_id_col].unique(), shuffle=True, test_size=0.2, random_state=RANDOM_STATE)
```

```python Collapsed="false"
EVENT_SUFFIX = ''

def split(df):
    X, y, groups = split_data(
        df=df,
        feature_cols=feature_cols,
        target_col=f'Event{EVENT_SUFFIX}',
        subject_id_col=subject_id_col,
        series_id_col=f'SeriesId{EVENT_SUFFIX}',
    )
    return X, y, groups
```

```python Collapsed="false"
df = data_feature_extracted.copy()

data_train = df[df[subject_id_col].isin(subjects_train)]
data_test = df[df[subject_id_col].isin(subjects_test)]

X_train, y_train, groups_train = split(data_train)
X_test, y_test, groups_test = split(data_test)
```

```python Collapsed="false"
from pmlib.utils import print_data_info

print(f'Training set:')
print_data_info(X_train, y_train, groups_train)
print(f'Testing set:')
print_data_info(X_test, y_test, groups_test)
```

<!-- #region Collapsed="false" -->
## Candidate Models Selection
<!-- #endregion -->

```python Collapsed="false"
monitoring_time = 24
warning_time = 8
```

<!-- #region Collapsed="false" -->
### Define Tested Models and Hyperparameters
<!-- #endregion -->

```python Collapsed="false"
est = XGBClassifier(n_jobs=1, random_state=42)

param_distributions = {
    'smoothing_window': [1, 3, 5, 7],
    'estimator__max_depth': [3, 4, 5, 6, 7],
    'estimator__n_estimators': [32, 64, 128, 256],
    'estimator__learning_rate': [0.05, 0.1, 0.15, 0.2],
    'estimator__booster': ['gbtree', 'dart'],
    'estimator__min_child_weight': [1,  4, 16, 64],
    'estimator__subsample': [0.6, 0.7, 0.8, 0.9, 1],
    'estimator__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1]
}
```

<!-- #region Collapsed="false" -->
### Define Evaluation Metrics
<!-- #endregion -->

```python Collapsed="false"
scorers = {
     "AUPRG": make_scorer(
        score_func=series_scorer(
            score_func=auprg_score,
            artificial_window=monitoring_time,
            warning_time=warning_time
        ),
        needs_proba=True
    ),
    "classical F1": make_scorer(
        score_func=series_scorer(
            score_func=f1_score,
            artificial_window=monitoring_time,
            warning_time=warning_time
        )
    ),
    'event-based F1': make_scorer(
        score_func=f1_reduced_score,
        monitoring_time=monitoring_time,
        warning_time=warning_time,
        alpha=0.8,
    ),
}
```

<!-- #region Collapsed="false" -->
### CV Search
<!-- #endregion -->

```python Collapsed="false"
search_pipeline = RandomizedSearchCV(
    estimator=PredictiveMaintenanceModel(
        estimator=est,
        artificial_window=monitoring_time
    ),
    param_distributions=param_distributions,
    scoring=scorers,
    cv=GroupKFold(n_splits=3),
    refit=False,
    verbose=10,
    n_jobs=N_JOBS,
    n_iter=N_JOBS*2,
    random_state=RANDOM_STATE
)

search_pipeline.fit(X_train, y_train, groups_train)

cv_results = pd.DataFrame(search_pipeline.cv_results_)
```

<!-- #region Collapsed="false" -->
### Pair plot
<!-- #endregion -->

```python Collapsed="false"
scorer_columns = [f'mean_test_{scorer_name}' for scorer_name in scorers.keys()]
scorer_rank_columns = [f'rank_test_{scorer_name}' for scorer_name in scorers.keys()]
param_cols = cv_results.columns[cv_results.columns.str.startswith('param_')]
```

```python Collapsed="false"
def correlation_analysis(df):
    df = df.copy()
    df = df.rename(columns={s: f'rank by {s[10:]}' for s in df.columns if s != 'accuracy'})
    g = sns.pairplot(df, diag_kind=None)
    plt.savefig('./images/azure/experiments_failure_prediction_azure_correlation.pdf', bbox_inches='tight')
    plt.show()

correlation_analysis(cv_results[scorer_rank_columns])
```

<!-- #region Collapsed="false" -->
# Candidate Models Comparison
<!-- #endregion -->

```python Collapsed="false"
df = (
    cv_results
    .copy()
    .loc[lambda df: df[scorer_rank_columns].min(axis=1).eq(1),
         [*scorer_rank_columns, *param_cols, *scorer_columns]]
    .reset_index(drop=True)
)
df[scorer_columns] = df[scorer_columns].astype(float).round(10)
df = df.rename(columns={s: f'rank by {s.upper()[10:]}' for s in scorer_rank_columns})
df = df.rename(columns={s: f'{s.upper()[10:]}' for s in scorer_columns})
df = df.T
df.columns = list(string.ascii_uppercase[:df.columns.shape[0]])
df.to_latex('./images/azure/candidate_models.tex')
display(df)

candidate_models_parameters = df
```

<!-- #region Collapsed="false" -->
## Train the Candidate Models on Full Training Set
<!-- #endregion -->

```python Collapsed="false"
model_ids = ['AUPRG', 'event-based F1', 'classical F1']
```

```python Collapsed="false"
candidate_models = []
# for i in range(len(selected_models_parameters.columns)):
for i in range(len(candidate_models_parameters.columns)):
    params = candidate_models_parameters.T[param_cols]
    params.columns = params.columns.str[6:]
    params = params.iloc[i].to_dict()
    print(params)
    model = PredictiveMaintenanceModel(
        estimator=XGBClassifier(n_jobs=N_JOBS, random_state=1),
        artificial_window=monitoring_time
    )
    model.set_params(**params)
    model.fit(X_train, y_train)
    candidate_models.append(model)
candidate_models
```

<!-- #region Collapsed="false" -->
## Calculate PR Curves
<!-- #endregion -->

```python Collapsed="false"
%%time
models_pr_curves = []

alpha = 0.8

for model in candidate_models:

    y_true_series = y_test
    y_score_series = model.predict_proba(X_test)

    thresholds = np.sort(np.unique(np.hstack(y_score_series)))
    # There are lots of predictions with low scores - omit low thresholds
    thresholds = thresholds[thresholds >= 0.01]

    def get_pred_from_score(y_score_series, threshold):
        y_pred_series = [
                np.array(y_score) >= threshold
            for y_score in y_score_series
        ]
        return y_pred_series
          
    def f(threshold):
        y_pred_series = get_pred_from_score(y_score_series, threshold)
        recall = recall_reduced_score(y_true_series, y_pred_series, monitoring_time, warning_time, alpha)
        precision = precision_reduced_score(y_true_series, y_pred_series, monitoring_time, warning_time, alpha)
        return [recall, precision]
    
    pool = multiprocessing.Pool(N_JOBS)

    results = pool.map(f, tqdm(thresholds))
    pool.close()
    recalls, precisions = np.array(results).T
    
    models_pr_curves.append((precisions, recalls, thresholds))
```

```python Collapsed="false"
plt.figure(figsize=(6, 4))
for metric, (precisions, recalls, thresholds) in zip(model_ids, models_pr_curves):
    plt.plot(recalls, precisions, label=metric, alpha=0.8)
plt.xlabel('event-based recall')
plt.ylabel('event-based precision')
plt.ylim(0.58, 1.02)
plt.xlim(0.58, 1.02)
plt.legend(title='Model selected by:')
plt.grid()
plt.savefig('./images/azure/experiments_failure_prediction_azure_pr_curves_reduced.pdf', bbox_inches='tight')
```

```python Collapsed="false"
%%time
models_classical_pr_curves = []

for model in candidate_models:
    y_true_series = y_test
    y_score_series = model.predict_proba(X_test)
    (precisions, recalls, thresholds) = series_scorer(precision_recall_curve, monitoring_time, warning_time)(y_true_series, y_score_series)
    models_classical_pr_curves.append([precisions, recalls, thresholds])
```

```python Collapsed="false"
plt.figure(figsize=(6, 4))
for model_id, (precisions, recalls, thresholds) in zip(model_ids, models_classical_pr_curves):
    mask = recalls < 2
    plt.plot(recalls[mask], precisions[mask], label=model_id)
plt.xlabel('classical recall')
plt.ylabel('classical precision')
plt.ylim(0.58, 1.02)
plt.xlim(0.58, 1.02)
plt.grid()
plt.legend(title='Model selected by:')
plt.savefig('./images/azure/experiments_failure_prediction_azure_pr_curves_ts.pdf', bbox_inches='tight')
```

```python Collapsed="false"
def calc_f1(prec, rec):
    return 2*prec*rec / (prec + rec)

for (model_id,
    (precisions, recalls, thresholds),
    (reduced_precisions, reduced_recalls, reduced_threhsolds),
    (existence_precisions, existence_recalls, existence_threhsolds),
) in list(zip(
        model_ids,
        models_classical_pr_curves,
        models_pr_curves,
        models_existence_pr_curves,
)):
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(np.r_[thresholds, 1], recalls, label='classical recall', color='red', alpha=0.3)
    plt.plot(np.r_[thresholds, 1], precisions, label='classical precision', color='green', alpha=0.3)
    plt.plot(np.r_[thresholds, 1], calc_f1(precisions, recalls), label='classical F1', color='purple', alpha=0.3)
    plt.plot(reduced_threhsolds, reduced_recalls, label='event-based recall', color='red')
    plt.plot(reduced_threhsolds, reduced_precisions, label='event-based precision', color='green')
    plt.plot(reduced_threhsolds, calc_f1(reduced_precisions, reduced_recalls), label='event-based F1', color='purple')
    plt.xlabel('decision threshold')
    print(f'Model selected by {model_id}')
    plt.grid()
    plt.ylim(0.85, 1.01)
    
    ax.figure.legend(loc='upper left', bbox_to_anchor=(0.35, 0.35))
    plt.savefig(f'./images/azure/experiments_failure_prediction_azure_multicurve_{model_id}.pdf', bbox_inches='tight')
    plt.show()
```
