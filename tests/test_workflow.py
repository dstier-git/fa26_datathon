import importlib
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from workplace_reviews.data import load
from workplace_reviews.modeling import out_of_fold_scores
from workplace_reviews.pipeline import run


def fixture_data():
    rows = []
    for company in range(15):
        for rating in [1, 2, 3, 4, 5] * 2:
            rows.append(dict(company_name=f"employer{company}", rating_overall=rating,
                             review_pros="good team benefits" if rating > 2 else "none toxic terrible",
                             review_cons="flexible supportive" if rating > 2 else "awful management hostile"))
    return pd.DataFrame(rows)


def test_import_and_help(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    importlib.reload(importlib.import_module("workplace_reviews.pipeline"))
    result = subprocess.run([sys.executable, "-m", "workplace_reviews", "--help"], capture_output=True)
    assert result.returncode == 0
    assert not list(tmp_path.iterdir())


def test_loader_excludes_samples(tmp_path):
    pd.DataFrame({"value": [1]}).to_csv(tmp_path / "full.csv", index=False)
    pd.DataFrame({"value": [99]}).to_csv(tmp_path / "sample.csv", index=False)
    assert load(str(tmp_path))["value"].tolist() == [1]


def test_company_splits_and_oof(monkeypatch):
    from workplace_reviews import modeling
    gd = fixture_data().rename(columns={"company_name": "co"})
    gd["y"] = (gd.rating_overall <= 2).astype(int)
    gd["text"] = gd.review_pros + " " + gd.review_cons
    tr, te = next(GroupShuffleSplit(test_size=.2, random_state=42).split(gd, groups=gd.co))
    assert set(gd.co.iloc[tr]).isdisjoint(gd.co.iloc[te])
    original = modeling.GroupKFold
    visits = np.zeros(len(gd))
    class CheckedFolds:
        def __init__(self, **kwargs):
            self.folds = original(**kwargs)
        def split(self, frame, y, groups):
            for train, test in self.folds.split(frame, y, groups):
                assert set(groups.iloc[train]).isdisjoint(groups.iloc[test])
                visits[test] += 1
                yield train, test
    monkeypatch.setattr(modeling, "GroupKFold", CheckedFolds)
    scores = out_of_fold_scores(gd)
    assert np.all(visits == 1)
    assert np.isfinite(scores).all()
    assert ((scores >= 0) & (scores <= 1)).all()


def test_end_to_end(tmp_path, monkeypatch):
    gd_dir, in_dir = tmp_path / "gd", tmp_path / "indeed"
    gd_dir.mkdir(); in_dir.mkdir()
    fixture_data().to_csv(gd_dir / "reviews.csv", index=False)
    pd.DataFrame([dict(company_name=f"employer{i}", job_title=f"job{j}", location="Oakland",
                       salary_formatted="$20 an hour", company_rating=4) for i in range(15) for j in range(3)]).to_csv(in_dir / "jobs.csv", index=False)
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "nested" / "out"
    run(gd_dir, in_dir, output)
    assert {p.name for p in output.iterdir()} == {"RESULTS.md", "failures.csv", "company_join.csv"}
    joined = pd.read_csv(output / "company_join.csv")
    assert len(joined) == 15
    assert np.allclose(joined.actual_neg_share, .4)
    assert (joined.reviews == 10).all()
    assert (joined.postings == 3).all()
    assert not (tmp_path / "RESULTS.md").exists()
    assert "PROS:" not in (output / "RESULTS.md").read_text()
