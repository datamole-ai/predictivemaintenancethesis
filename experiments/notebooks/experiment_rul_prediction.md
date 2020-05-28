---
jupyter:
  jupytext:
    formats: ipynb,md
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
# Remaining Useful Life Prediction of Turbofan Engines
<!-- #endregion -->

```python Collapsed="false"
# Enable autoreloading
%reload_ext autoreload
%autoreload 2

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
from pmlib.data_transformation import split_data
from pmlib.preprocessing import create_series_id
from pmlib.modeling import smooth_predictions
from sklearn.preprocessing import RobustScaler

%matplotlib inline
# mpl.style.use('ggplot')
```

```python Collapsed="false"
N_JOBS = 32
RANDOM_STATE = 7
```

```python Collapsed="false"
!mkdir -p ./data/turbofan_engine_degradation/
!mkdir -p ./images/turbofan_engine_degradation/
```

<!-- #region Collapsed="false" -->
## Load and Preprocess Data
<!-- #endregion -->

```python Collapsed="false"
!wget -nc https://ti.arc.nasa.gov/m/project/prognostic-repository/CMAPSSData.zip ./data/turbofan_engine_degradation/
!unzip -o ./data/turbofan_engine_degradation/CMAPSSData.zip -d ./data/turbofan_engine_degradation/
```

```python Collapsed="false"
RAW_DATA_PATH = Path('./data/turbofan_engine_degradation/train_FD001.txt')
```

```python Collapsed="false"
sensor_cols = [f'sensor_{i}' for i in range(1, 27)]
operational_setting_cols = [f'operational_setting_{i}' for i in range(1, 4)]
feature_cols = [*operational_setting_cols, *sensor_cols]
columns = ['unit number', 'cycles', *feature_cols]

data = pd.read_csv(RAW_DATA_PATH, sep=' ', header=None, names=columns)

data.dropna(axis=1, inplace=True)

sensor_cols = data.columns[data.columns.str.startswith('sensor_')]
operational_setting_cols = data.columns[data.columns.str.startswith('operational_setting_')]
feature_cols = [*operational_setting_cols, *sensor_cols]
columns = ['unit number', 'cycles', *feature_cols]

def f(df):
    df[feature_cols] = df[feature_cols].rolling(11, min_periods=1).mean()
    return df

data = data.groupby('unit number').apply(f)

# Drop columns with zero variance
data = data.drop(columns=data.columns[data.var() == 0])
sensor_cols = data.columns[data.columns.str.startswith('sensor_')]
operational_setting_cols = data.columns[data.columns.str.startswith('operational_setting_')]

feature_cols = [*operational_setting_cols, *sensor_cols, 'cycles']

data['RUL'] = (1 + data.iloc[::-1].groupby('unit number').cumcount().iloc[::-1])
```

```python Collapsed="false"
subject_id_col = 'unit number'
```

```python Collapsed="false"
data.info()
```

```python Collapsed="false"
data.apply(np.round, decimals=2)
```

```python Collapsed="false"
df = data.loc[lambda df: df['unit number'].eq(1), sensor_cols]
df.index += 1
df[df.columns] = RobustScaler().fit_transform(df)

df[df.columns[[1, 2, 3, 8]]].plot(figsize=(10, 4))
plt.grid()
plt.xlabel('cycles')
plt.savefig('./images/turbofan_engine_degradation/experiments_rul_sensor_values.pdf', bbox_inches='tight')
```

<!-- #region Collapsed="false" -->
## Data Splitting
<!-- #endregion -->

```python Collapsed="false"
from pmlib.data_transformation import split_data
from pmlib.utils import print_data_info
from sklearn.model_selection import train_test_split


def split_turbofan_data(df):
    X, y, groups = split_data(df=df,
                              feature_cols=feature_cols,
                              target_col='RUL',
                              series_id_col='unit number',
                              subject_id_col='unit number',
                             )
    return X, y, groups



subjects_train, subjects_test = train_test_split(data[subject_id_col].unique(), shuffle=True, test_size=0.2, random_state=1)

data_train = data.loc[data[subject_id_col].isin(subjects_train)]
data_test = data.loc[data[subject_id_col].isin(subjects_test)]

X_train, y_train, groups_train = split_turbofan_data(data_train)
X_test, y_test, groups_test = split_turbofan_data(data_test)

print('Training set:')
print_data_info(X_train, y_train, groups_train)
print('Testing set:')
print_data_info(X_test, y_test, groups_test)
```

<!-- #region Collapsed="false" -->
## Candidate Models Selection
<!-- #endregion -->

```python Collapsed="false"
warning_time = 5
```

```python Collapsed="false"
from itertools import product
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

tested_models = [
    *[
        XGBRegressor(max_depth=max_depth, n_estimators=n_estimators, n_jobs=1, random_state=5)
        for max_depth, n_estimators in product([3, 4, 5, 7, 8, 9], [16, 32, 64, 128, 256])
    ],
    *[
        SVR(kernel='rbf', C=C, gamma=gamma)
        for gamma, C in product([0.01, 0.1, 0.5, 1], [0.1, 1, 10, 100])
    ],
]

tested_models = [Pipeline([
    ('scaler', RobustScaler(quantile_range=(5, 90))),
    ('estimator', estimator)
]) for estimator in tested_models]

param_distributions = {
    'estimator': tested_models,
}
```

<!-- #region Collapsed="false" -->
### Define Evaluation Metrics
<!-- #endregion -->

```python Collapsed="false"
from pmlib.evaluation.rul import (mean_prognostic_horizon,
                                  mean_absolute_percentage_error,
                                  mean_asymmetrically_weighted_percentage_error)
from sklearn.metrics import (mean_absolute_error,
                             mean_squared_error,
                             mean_squared_log_error)
from sklearn.metrics import make_scorer
from pmlib.evaluation.utils import series_scorer

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

scorers = {
    "rmse": make_scorer(
        score_func=series_scorer(
            root_mean_squared_error,
            warning_time=warning_time
        ),
        greater_is_better=False
    ),
    "mape": make_scorer(
        score_func=series_scorer(
            mean_absolute_percentage_error,
            warning_time=warning_time
        ),
        greater_is_better=False
    ),
    'mape@40': make_scorer(
        score_func=series_scorer(
            score_func=mean_absolute_percentage_error,
            artificial_window=1,
            warning_time=warning_time,
            evaluation_window=40
        ),
        greater_is_better=False
    ),
    'mph_10': make_scorer(
        score_func=mean_prognostic_horizon,
        alpha=10
    ),
}
```

<!-- #region Collapsed="false" -->
### CV Search
<!-- #endregion -->

```python Collapsed="false"
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from pmlib.modeling import PredictiveMaintenanceModel

search_pipeline = RandomizedSearchCV(
    estimator=PredictiveMaintenanceModel(),
    param_distributions=param_distributions,
    scoring=scorers,
    cv=GroupKFold(n_splits=10),
    refit=False,
    verbose=10,
    n_jobs=N_JOBS,
    n_iter=100,
    random_state=5,
)

training_series_limit = int(1e10)
# training_series_limit = 30
search_pipeline.fit(X_train[:training_series_limit],
                    y_train[:training_series_limit],
                    groups_train[:training_series_limit])

cv_results = pd.DataFrame(search_pipeline.cv_results_)
```

<!-- #region Collapsed="false" -->
### Correlation Analysis of Evaluation Metrics
<!-- #endregion -->

```python Collapsed="false"
scorer_columns = [f'mean_test_{scorer_name}' for scorer_name in scorers.keys()]
scorer_rank_columns = [f'rank_test_{scorer_name}' for scorer_name in scorers.keys()]
param_cols = cv_results.columns[cv_results.columns.str.startswith('param_')]
```

```python Collapsed="false"
import seaborn as sns

df = cv_results.copy()

renaming = {s: f'{s.upper()[10:]}' for s in scorer_rank_columns}
df = df[scorer_rank_columns].rename(columns=renaming).corr()
sns.heatmap(df, annot=True, cmap='Greens', vmin=-1, vmax=1)
plt.savefig('./images/turbofan_engine_degradation/experiments_rul_heatmap.pdf', bbox_inches='tight')
```

```python Collapsed="false"
cols = [f'rank_test_{col}' for col in ['rmse', 'mape', 'mape@40', 'mph_10']]

renaming = {s: f'{s.upper()[10:]} rank' for s in cols}

df = cv_results.rename(columns=renaming).copy()
df['param_estimator'] = df['param_estimator'].apply(lambda x: type(x[-1]).__name__)
rank_cols = list(renaming.values())
plt.figure(figsize=(3, 3))
sns.pairplot(df, x_vars=rank_cols, y_vars=rank_cols,
#              hue='param_estimator',
             diag_kind=None)
plt.savefig('./images/turbofan_engine_degradation/experiments_rul_pairplot.pdf', bbox_inches='tight')
```

```python Collapsed="false"
import string


def extract_params(x):
    model = x[-1]
    all_params = model.get_params()
    params = dict()
    if type(model) == XGBRegressor:
        params = {k:all_params[k] for k in ('n_estimators', 'max_depth')}
    elif type(model) == SVR:
        params = {k:all_params[k] for k in ('C', 'gamma')}
    return params


df = (
    cv_results
    .copy()
    .loc[lambda df:
             df[scorer_rank_columns].min(axis=1).eq(1)
         ,[*scorer_rank_columns, *param_cols, *scorer_columns]]
    .reset_index(drop=True)
    .drop_duplicates(subset=scorer_rank_columns)
)
df[scorer_columns] =  df[scorer_columns].astype(float).abs().round(1)
df = df.rename(columns={s: f'rank by {s.upper()[10:]}' for s in scorer_rank_columns})
df = df.rename(columns={s: f'{s.upper()[10:]}' for s in scorer_columns})
df['params'] = df['param_estimator'].apply(extract_params)
df = df.T
df.loc['regressor'] = df.loc['param_estimator']
df.loc['regressor'] = df.loc['regressor'].apply(lambda x: type(x[-1]).__name__)
# df.drop(labels=['param_estimator']).to_latex('candidate_models.tex')
display(df.drop(labels=['param_estimator']))

selected_models_parameters = df
```

```python Collapsed="false"
df.loc['param_estimator'].values[0]
```

```python Collapsed="false"
from tqdm import tqdm


def train_selected_pdm_model(estimator):
    pdm_model = PredictiveMaintenanceModel(estimator=estimator)
    try:
        pdm_model.estimator.set_params(n_jobs=N_JOBS)
    except Exception:
        pass
    try:
        pdm_model.estimator.set_params(random_state=RANDOM_STATE)
    except Exception:
        pass
    pdm_model.fit(X_train, y_train)
    return pdm_model


selected_models = [train_selected_pdm_model(model) for model in tqdm(df.loc['param_estimator'].values)]
```

<!-- #region Collapsed="false" -->
## Candidate Models Comparison
<!-- #endregion -->

```python Collapsed="false"
candidate_models_predictions = [model.predict(X_test) for model in tqdm(selected_models)]
candidate_models_predictions = np.array(list(zip(['XGBoost', 'SVR'], np.array(candidate_models_predictions))))
```

```python Collapsed="false"
def relative_mape(y, y_pred):
    return np.abs((y - y_pred) / y) * 100

fig, ax = plt.subplots(figsize=(8, 5))
for model_id, y_pred in candidate_models_predictions:
    df = pd.DataFrame({'RUL': np.hstack(y_test),
                       'MAPE': relative_mape(np.hstack(y_test), np.hstack(y_pred))})

    df = (
        df
        .loc[df['RUL'].gt(warning_time)
            ]
        .reset_index()
    )
    sns.lineplot(data=df, x='RUL', y='MAPE', label=model_id)
plt.xlim(225, -5)
plt.ylim(0, 100)
plt.legend(title='Model')
plt.xlabel('actual RUL')
plt.ylabel('MAPE')
plt.grid()
# plt.axvline(warning_time, color='black')
plt.savefig('./images/turbofan_engine_degradation/experiments_rul_relative_mape.pdf', bbox_inches='tight')
```

```python Collapsed="false"
rul_lim = 150

for i in [5, 12, 15, 19]:
    subject_id = subjects_test[i]
    plt.figure(figsize=(4, 3))
    for model_id, y_pred_series in candidate_models_predictions:
        preds = y_pred_series[i][-rul_lim:][::-1]
        plt.plot(preds, label=f'{model_id} prediction')
    plt.plot(y_test[i][-rul_lim:][::-1], color='black', linestyle='--', label='true RUL')
    plt.grid()
    plt.xlabel('true RUL')
    plt.ylabel('predicted RUL')
    plt.xlim(155, -5)
    plt.legend()
    plt.title(f'Subject {subject_id}')
    plt.savefig(f'./images/turbofan_engine_degradation/experiments_rul_prediction_{subject_id}.pdf', bbox_inches='tight')
    plt.show()
```

<!-- #region Collapsed="false" -->
## Miscellaneous
<!-- #endregion -->

```python Collapsed="false"
rul_lim = 250

i = 6
subject_id = subjects_test[i]
model_id, y_pred_series = the_models[0]
predicted = y_pred_series[i][-rul_lim:]
actual = y_test[i][-rul_lim:]

plt.figure(figsize=(8, 4))

plt.plot(predicted, label=f'predicted RUL')
plt.plot(actual, color='black', linestyle='--', label='actual RUL')
plt.fill_between(range(len(predicted)), actual, np.max(predicted), alpha=0.2, color='red', label='late (optimistic) predictions')
plt.fill_between(range(len(predicted)), actual, 0, alpha=0.2, color='green', label='early (pessimistic) predictions')
plt.grid()
plt.xlabel('t')
# plt.xlim(-5, 155)
plt.legend()
plt.savefig(f'./images/turbofan_engine_degradation/approaches_rul_prediction_optimistic.pdf', bbox_inches='tight')
plt.show()
```
