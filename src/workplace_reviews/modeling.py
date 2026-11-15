"""Fixed text model and company-grouped out-of-fold scoring."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

def make_model():
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000,
                          sublinear_tf=True, strip_accents="unicode")
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, solver="liblinear")
    return vec, clf


def out_of_fold_scores(gd):
    scores = np.full(len(gd), np.nan)
    for tr, te in GroupKFold(n_splits=3).split(gd, gd["y"], groups=gd["co"]):
        vec, clf = make_model()
        clf.fit(vec.fit_transform(gd["text"].iloc[tr]), gd["y"].iloc[tr])
        scores[te] = clf.predict_proba(vec.transform(gd["text"].iloc[te]))[:, 1]
    return scores
