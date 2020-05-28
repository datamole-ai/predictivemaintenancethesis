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
# Fault Detection of APS Scania Trucks

Outline:

1. [Data Loading](#Data-Loading)
1. [Modeling](#Modeling)
1. [Evaluation](#Evaluation)
<!-- #endregion -->

```python Collapsed="false"
# Enable autoreloading
%reload_ext autoreload
%autoreload 2
```

```python Collapsed="false" toc-hr-collapsed=true toc-nb-collapsed=true Collapsed="false"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score,
                             f1_score,
                             make_scorer,
                             precision_recall_curve,
                             plot_precision_recall_curve,
                             roc_auc_score)
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import plot_precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from urllib.parse import urljoin
from urllib.request import urlopen
from xgboost import XGBClassifier

from pmlib.evaluation.classical import aupr_score, auprg_score
```

```python Collapsed="false"
N_JOBS = 32
RANDOM_STATE = 7
```

```python Collapsed="false"
!mkdir -p ./images/scania/
```

<!-- #region Collapsed="true" toc-hr-collapsed=true toc-nb-collapsed=true Collapsed="false" -->
# Data Loading
<!-- #endregion -->

<!-- #region Collapsed="false" -->
The data set consits of 76K rows and 171 columns.
Each row represent data about one APS Scania truck where:
 - column 'class' contains a binary variable whether a failure occured;
 - the rest of the columns represent 170 anonymized features.

The amount of positive classes in the data set is 1375 (1.81 % of total samples).
<!-- #endregion -->

```python Collapsed="false"
def load_aps_scania_trucks_data_set():

    base_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00421/'
    
    def load_data_csv(file_name):
        data = pd.read_csv(base_url + file_name, skiprows=20, na_values="na")
        data['class'] = data['class'].replace({'neg': 0, 'pos': 1}).astype(int)
        return data
    
    data_train = load_data_csv('aps_failure_training_set.csv')
    data_test = load_data_csv('aps_failure_test_set.csv')
    
    return data_train, data_test
```

```python Collapsed="false"
data_train, data_test = load_aps_scania_trucks_data_set()

target_col = 'class'
feature_cols = data_train.drop(columns=[target_col]).columns
```

```python Collapsed="false"
data_train.info()
```

```python Collapsed="false"
data_train.sample(10)
```

```python Collapsed="false"
data_train.isna().sum().sort_values(ascending=False).head(10)
```

```python Collapsed="false"
for df in [data_train, data_test]:
    samples = df.shape[0]
    pos = df["class"].sum()
    pos_pctg = np.round(100 * pos / samples, 2)

    print(f'Dataset size: {samples} samples')
    print(f'Positive classes: {pos} ({pos_pctg} %)')
```

<!-- #region Collapsed="false" toc-hr-collapsed=true toc-nb-collapsed=true Collapsed="false" -->
## Data Splitting
<!-- #endregion -->

```python Collapsed="false"
data_train_cv, data_train_val = train_test_split(data_train, test_size=0.2, stratify=data_train['class'], shuffle=True, random_state=RANDOM_STATE)

X_train, y_train = data_train_cv[feature_cols].values, data_train_cv['class'].values
X_val, y_val = data_train_val[feature_cols].values, data_train_val['class'].values
X_test, y_test = data_test[feature_cols].values, data_test['class'].values
```

<!-- #region Collapsed="true" toc-hr-collapsed=true toc-nb-collapsed=true Collapsed="false" -->
## Candidate Models Selection
<!-- #endregion -->

<!-- #region Collapsed="false" toc-hr-collapsed=true toc-nb-collapsed=true Collapsed="false" -->
### Hyperparameter Search CV
<!-- #endregion -->

<!-- #region Collapsed="false" toc-hr-collapsed=true toc-nb-collapsed=true Collapsed="true" -->
Pipeline:
 - Imputing
 - PCA
 - balancing:
 
3 folds

Random search of 42 iterations
<!-- #endregion -->

<!-- #region Collapsed="false" -->
#### Pipeline
<!-- #endregion -->

```python Collapsed="false"
estimator = Pipeline([
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
    ('clf', XGBClassifier(random_state=10)),
])

param_grid = {
    'clf__max_depth': [2, 3, 4, 5, 6, 7],
    'clf__n_estimators': [4, 8, 16, 32, 64, 128, 256],
    'clf__learning_rate': [0.05, 0.1, 0.15, 0.2],
    'clf__booster': ['gbtree', 'dart'],
    'clf__min_child_weight': [ 1,  4, 16, 64],
    'clf__subsample': [0.6, 0.7, 0.8, 0.9, 1],
    'clf__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1]
}
```

<!-- #region Collapsed="false" -->
#### Scorers
<!-- #endregion -->

```python Collapsed="false"
def f1_at_fpr(y_true, y_score, fpr=0.05):
    fprs, tprs, thresholds = roc_curve(y_true, y_score)
    threshold = np.interp(fpr, fprs, thresholds)
    y_pred = y_score >= threshold
    return fbeta_score(y_true, y_pred, beta=5)

def auc_at_fpr(y_true, y_score, fpr=0.05):
    fprs, tprs, thresholds = roc_curve(y_true, y_score)
    threshold = np.interp(fpr, fprs, thresholds)
    mask = thresholds <= threshold
    return auc(fprs[mask], tprs[mask])


scorers = {
    'auprg': make_scorer(auprg_score, needs_proba=True),
    'f1': make_scorer(f1_score),
    'accuracy': make_scorer(accuracy_score),
    'auroc': make_scorer(roc_auc_score, needs_proba=True),
}
```

<!-- #region Collapsed="false" -->
#### Run
<!-- #endregion -->

```python Collapsed="false"
%%time
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

search_pipeline = RandomizedSearchCV(estimator, param_grid,
                                     scoring=scorers,
                                     n_jobs=N_JOBS,
                                     verbose=10,
                                     cv=cv,
                                     refit=False,
                                     n_iter=N_JOBS*5,
                                     random_state=RANDOM_STATE
                                    )

search_pipeline.fit(X_train, y_train)
```

<!-- #region Collapsed="false" -->
### Analyze Results and Select Candidate Models
<!-- #endregion -->

```python Collapsed="false"
cv_results = pd.DataFrame(search_pipeline.cv_results_)
cv_results['param_clf__max_depth'] = cv_results['param_clf__max_depth'].astype(int)
cv_results['param_clf__n_estimators'] = cv_results['param_clf__n_estimators'].astype(int)
```

<!-- #region Collapsed="false" -->
#### Correlation Between Ranks of Models by Metrics
<!-- #endregion -->

```python Collapsed="false"
cv_results.sort_values('mean_fit_time', ascending=False).head()
```

```python Collapsed="false"
param_cols = cv_results.columns[cv_results.columns.str.startswith('param_')]
mean_test_cols = cv_results.columns[cv_results.columns.str.startswith('mean_test_')]
rank_test_cols = cv_results.columns[cv_results.columns.str.startswith('rank_test_')]
```

```python Collapsed="false"
for metric in ['f1', 'auroc']:
    print(f'Metric: {metric}')
    print(f'Params: {cv_results.sort_values(f"mean_test_{metric}").iloc[-1][param_cols].to_dict()}')
```

```python Collapsed="false"
df = cv_results

sns.heatmap(df[rank_test_cols].corr(), annot=True, cmap='PRGn', vmin=-1, vmax=1)
```

```python Collapsed="false"
df = cv_results[rank_test_cols].drop(columns=['rank_test_aupr']).copy()
df = df.rename(columns={s: f'rank by {s.upper()[10:]}' for s in df.columns if s != 'accuracy'})
sns.pairplot(df)
plt.savefig('./images/scania/experiments_fault_detection_aps_pairplot.pdf', bbox_inches='tight')
```

```python Collapsed="false"
import string

df = (
    cv_results
    .copy()
    .loc[lambda df: df[rank_test_cols].min(axis=1).le(1),
         [*rank_test_cols, *param_cols, *mean_test_cols]]
    .reset_index(drop=True)
)
df[mean_test_cols] = df[mean_test_cols].astype(float).round(10)
df = df.rename(columns={s: f'rank by {s.upper()[10:]}' for s in rank_test_cols})
df = df.rename(columns={s: f'{s.upper()[10:]}' for s in mean_test_cols})
df = df.T
df.columns = list(string.ascii_uppercase[:df.columns.shape[0]])
df.to_latex('./images/scania/candidate_models.tex')
display(df)

selected_models_parameters = df
```

```python Collapsed="false"
models = dict()
for metric in ['auroc', 'f1']:
    df = cv_results.loc[cv_results[f'rank_test_{metric}'].eq(1), param_cols]
    df.columns = df.columns.str[11:]
    model_params = df.iloc[0].to_dict()
    model = Pipeline([
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
        ('clf', XGBClassifier(**model_params, n_jobs=N_JOBS, random_state=RANDOM_STATE)),
    ])
    model.fit(X_train, y_train)
    models[metric] = model
models
```

<!-- #region Collapsed="false" -->
## Final Model and Decision Threshold Selection

From the domain, we have a cost function for FPs and FNs:
  - \\$10 per FP
  - \\$500 per FN
 
The goal is to minimize the costs. Therefore, we select the threshold where the minimal cost is.
<!-- #endregion -->

```python Collapsed="false"
def plot_precision_recall_costs(y_true, y_proba,
                                fp_cost=10, fn_cost=500,
                                log_scale_threshold=True,
                                log_scale_cost=True
                               ):
    
    def cost_function(y_true, y_pred):
        fp = np.sum((y_true - y_pred) == -1)
        fn = np.sum((y_true - y_pred) == 1)
        return (fp * fp_cost + fn * fn_cost).astype(int)
    
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    thresholds = np.r_[thresholds, 1]

    costs = []
    fps = []
    fns = []
    
    for threshold in thresholds:
        y_pred = y_proba >= threshold
        fp = np.sum((y_true - y_pred) == -1)
        fn = np.sum((y_true - y_pred) == 1)
        costs.append(cost_function(y_true, y_pred))

    best_index = np.argmin(costs)
    threshold = thresholds[best_index]
    cost = np.round(costs[best_index], 2)
    precision = np.round(precisions[best_index], 2)
    recall = np.round(recalls[best_index], 2)
    f1 = np.round(2*precision*recall / (precision + recall), 2)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    xlabel = 'Threshold'
    if log_scale_threshold:
        xlabel += ' (log-scale)'
        plt.xscale('log')
    plt.xlabel(xlabel)

    lns1 = ax1.plot(thresholds, recalls, color='blue', label='Recall')
    lns2 = ax1.plot(thresholds, precisions, color='green', label='Precision')
    
    ax2 = ax1.twinx()
    lns4 = ax2.plot(thresholds, costs, color='orange', label='Cost')    
    cost_label = 'Cost'
    if log_scale_cost:
        ax2.set_yscale('log')
        cost_label += ' (log-scale)'
    ax2.set_ylabel(cost_label)
    
    # Annotate
    ax2.annotate(xy=(threshold, cost),
                 xytext=(threshold, cost*2),
                 arrowprops=dict(
                     facecolor='black',
                     shrink=0.05,
                     headwidth=10,
                     width=2,
                 ),
                 s=f'Lowest cost = {cost}\n'
                   f'Threshold = {threshold:.1e}\n'
                   f'Recall = {recall}\n'
                   f'Precision = {precision}'
            )
    ax1.set_ylabel('precision, recall')
    
    lns = lns1+lns2+lns4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)
    ax1.grid()
    return costs, thresholds, precisions, recalls

for metric in ['auroc', 'f1']:
    y_score = models[metric].predict_proba(X_val)[:, 1]
    costs, thresholds, precisions, recalls = \
        plot_precision_recall_costs(y_val, y_score, log_scale_threshold=True)
    best_threshold = thresholds[np.argmin(costs)]
    plt.savefig(f'./images/scania/experiments_fault_detection_aps_cost_threshold_{metric}.pdf', bbox_inches='tight')
```

```python Collapsed="false"
fig, ax = plt.subplots()
for metric in ['auroc', 'f1']:
    plot_precision_recall_curve(models[metric], X_val, y_val, ax=ax, label=metric)
```

<!-- #region Collapsed="false" -->
## Final Model Evaluation
<!-- #endregion -->

<!-- #region Collapsed="false" -->
Evaluation on the test set.
<!-- #endregion -->

```python Collapsed="false"
def cost_function(y_true, y_pred):
    fp = np.sum((y_true - y_pred) == -1)
    fn = np.sum((y_true - y_pred) == 1)
    return (fp * 10 + fn * 500).astype(int)

y_test_pred = models['auroc'].predict_proba(X_test)[:, 1] >= best_threshold

print(f'Accuracy: {np.round(accuracy_score(y_test, y_test_pred), 2)}')
print(f'Recall: {np.round(recall_score(y_test, y_test_pred), 2)}')
print(f'Precision: {np.round(precision_score(y_test, y_test_pred), 2)}')
print(f'F1 score: {np.round(f1_score(y_test, y_test_pred), 2)}')
print(f'Cost: {np.round(cost_function(y_test, y_test_pred), 2)}')
```
