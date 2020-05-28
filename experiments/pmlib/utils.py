import numpy as np


def print_data_info(X, y, groups):
    print(f'\t# of subjects: {len(np.unique(groups))}')
    print(f'\t# of series: {X.shape[0]}')
    print(f'\t# of samples: {np.concatenate(X).shape[0]}')
