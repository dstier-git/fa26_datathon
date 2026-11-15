"""Classification metrics and company correlations."""
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

def metrics(y, pred, prob=None):
    y, pred = np.asarray(y), np.asarray(pred)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "macro_F1": f1_score(y, pred, average="macro", zero_division=0),
        "precision_neg": precision_score(y, pred, zero_division=0),
        "recall_neg": recall_score(y, pred, zero_division=0),
        "false_pos_rate": fp / (fp + tn) if (fp + tn) else np.nan,
        "AUC": roc_auc_score(y, prob) if prob is not None and len(set(y)) > 1 else np.nan,
    }


def spear(df, a, b):
    d = df[[a, b]].dropna()
    if len(d) < 5:
        return "n/a (too few companies)"
    return f"Spearman rho = {d[a].corr(d[b], method='spearman'):.3f} (n = {len(d)} companies)"
