
import numpy as np
from sklearn.metrics import auc, precision_recall_curve
from prg import prg


def auprg_score(y_true, y_score):
    y_true, y_score = np.array(y_true), np.array(y_score)
    prg_curve = prg.create_prg_curve(y_true, y_score)
    auprg = prg.calc_auprg(prg_curve)
    return auprg


def aupr_score(y_true, y_score):
    y_true, y_score = np.array(y_true), np.array(y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    aupr = auc(np.r_[rec, 0], np.r_[prec, 1])
    return aupr
