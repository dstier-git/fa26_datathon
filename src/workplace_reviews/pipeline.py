"""Run the preserved experiment without import-time side effects."""
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from .data import load, find
from .aggregation import join_companies
from .preprocessing import norm_co, status_group, region, role_family
from .modeling import make_model, out_of_fold_scores
from .evaluation import metrics, spear
from .reporting import Report, table

SEED = 42


def run(glassdoor_dir, indeed_dir, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    GD_DIR, IN_DIR = str(glassdoor_dir), str(indeed_dir)
    report = Report()
    log = report.log
    gd = load(GD_DIR)
    C_CO = find(gd, "company_name", "company", "employer")
    C_PRO = find(gd, "review_pros", "pros")
    C_CON = find(gd, "review_cons", "cons")
    C_RAT = find(gd, "overall_rating", "rating_overall", "overall", "star_rating", "rating")
    C_STAT = find(gd, "employee_status", "employment_status", "status", required=False)
    C_LOC = find(gd, "employee_location", "location", required=False)
    C_TITLE = find(gd, "job_title", "employee_title", "title", "role", required=False)

    log("# Results: Predicting Negative Workplace Reviews + Glassdoor x Indeed Company Join")
    log()
    log("## Column mapping")
    log(f"Glassdoor: company={C_CO}, pros={C_PRO}, cons={C_CON}, rating={C_RAT}, "
        f"status={C_STAT}, location={C_LOC}, title={C_TITLE}")

    n_raw = len(gd)
    gd["rating_num"] = pd.to_numeric(gd[C_RAT].astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0],
                                     errors="coerce")
    gd = gd[gd["rating_num"].between(1, 5)].copy()
    gd["pros"] = gd[C_PRO].fillna("").astype(str)
    gd["cons"] = gd[C_CON].fillna("").astype(str)
    gd = gd[(gd["pros"].str.strip() != "") | (gd["cons"].str.strip() != "")].copy()
    gd["text"] = "PROS: " + gd["pros"] + " CONS: " + gd["cons"]
    gd["y"] = (gd["rating_num"] <= 2).astype(int)
    gd["co"] = gd[C_CO].map(norm_co)
    gd["status"] = gd[C_STAT].map(status_group) if C_STAT else "Unknown"
    gd["region"] = gd[C_LOC].map(region) if C_LOC else "Unknown"
    gd["role"] = gd[C_TITLE].map(role_family) if C_TITLE else "Unknown"
    gd = gd.reset_index(drop=True)

    log()
    log("## Data decisions")
    log(f"- Glassdoor rows loaded: {n_raw:,}; kept {len(gd):,} with a valid 1-5 rating and pros/cons text.")
    log(f"- Target: negative review = overall rating <= 2 (positive class rate {gd['y'].mean():.1%}). "
        "Labels come from reviewers' own star ratings, so no labels were invented.")
    log("- Model input: pros + cons text only (rating never used as a feature).")
    log("- Company names normalized (lowercase, punctuation and legal suffixes like Inc/LLC/Corp removed) "
        "for grouping and for the Indeed join.")
    log("- Train/test split is GROUPED BY COMPANY (80/20): no company appears in both, "
        "so the model can't memorize company-specific wording. The test set was never used for training or tuning "
        "(hyperparameters are fixed defaults).")

    # ---------------------------------------------------------------- split + model
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr_idx, te_idx = next(gss.split(gd, gd["y"], groups=gd["co"]))
    train, test = gd.iloc[tr_idx].copy(), gd.iloc[te_idx].copy()


    vec, clf = make_model()
    Xtr = vec.fit_transform(train["text"])
    Xte = vec.transform(test["text"])
    clf.fit(Xtr, train["y"])
    test["prob"] = clf.predict_proba(Xte)[:, 1]
    test["pred"] = (test["prob"] >= 0.5).astype(int)

    # baselines
    majority = int(train["y"].mean() >= 0.5)
    KW = re.compile(r"\b(?:toxic|terrible|horrible|awful|worst|hostile|harass\w*|discriminat\w*|retaliat\w*|"
                    r"burn(?:ed|t)?\s?out|micromanag\w*|underpaid|disrespect\w*|abusive|bull(?:y|ying|ied))\b", re.I)
    test["kw_pred"] = test["cons"].str.contains(KW).astype(int)

    res = pd.DataFrame({
        "Majority class": metrics(test["y"], np.full(len(test), majority)),
        "Keyword rule (cons)": metrics(test["y"], test["kw_pred"]),
        "TF-IDF + LogReg (ours)": metrics(test["y"], test["pred"], test["prob"]),
    }).T
    log()
    log(f"## Model vs. baselines (held-out test set: {len(test):,} reviews from "
        f"{test['co'].nunique():,} companies never seen in training)")
    log(table(res))

    # top features
    fn = np.array(vec.get_feature_names_out())
    coef = clf.coef_[0]
    log()
    log("Top words pushing toward NEGATIVE: " + ", ".join(fn[np.argsort(coef)[-20:][::-1]]))
    log()
    log("Top words pushing toward POSITIVE: " + ", ".join(fn[np.argsort(coef)[:20]]))

    # ---------------------------------------------------------------- subgroups
    log()
    log("## Error and fairness analysis (test set, groups with >= 50 reviews)")
    for g in ["status", "region", "role"]:
        rows = {}
        for val, sub in test.groupby(g):
            if len(sub) < 50:
                continue
            m = metrics(sub["y"], sub["pred"], sub["prob"])
            rows[val] = {"n": len(sub), "neg_rate": sub["y"].mean(), **m}
        if rows:
            log(f"### By {g}")
            log(table(pd.DataFrame(rows).T.sort_values("n", ascending=False)))

    # failure examples (no company / title / location, to protect reviewers)
    fp_ex = test[test["y"] == 0].nlargest(10, "prob")
    fn_ex = test[test["y"] == 1].nsmallest(10, "prob")
    fail = pd.concat([fp_ex.assign(error="false_positive"), fn_ex.assign(error="false_negative")])
    fail[["error", "rating_num", "prob", "pros", "cons"]].to_csv(output_dir / "failures.csv", index=False)
    log()
    log("## Where the model fails (full examples in failures.csv)")
    log("Review quotations are kept only in the local failures.csv; removing identifiers does not guarantee anonymity.")

    # ---------------------------------------------------------------- out-of-fold scores for every review
    gd["oof_prob"] = out_of_fold_scores(gd)

    # ================================================================ INDEED JOIN
    ind = load(IN_DIR)
    I_CO = find(ind, "company_name", "company", "employer")
    I_SAL = find(ind, "salary_formatted", "salary", required=False)
    I_RAT = find(ind, "company_rating", required=False)
    I_TITLE = find(ind, "job_title", "title", required=False)
    I_LOC = find(ind, "location", required=False)
    log()
    log("## Dataset join: Glassdoor x Indeed (by normalized company name)")
    log(f"Indeed columns: company={I_CO}, salary={I_SAL}, company_rating={I_RAT}")

    ind, joined, usable, n_ind_raw = join_companies(gd, ind, I_CO, I_SAL, I_RAT, I_TITLE, I_LOC)
    joined.to_csv(output_dir / "company_join.csv")

    log(f"- Indeed postings: {n_ind_raw:,} raw, {len(ind):,} after deduping on title+company+location.")
    log(f"- Companies matched across datasets: {len(joined):,}; with >=20 reviews and >=3 postings: {len(usable):,}.")
    log(f"- Matched companies cover {gd['co'].isin(joined.index).mean():.1%} of Glassdoor reviews "
        f"and {ind['co'].isin(joined.index).mean():.1%} of deduped Indeed postings.")


    log()
    log("Company-level results (companies with >=20 Glassdoor reviews and >=3 Indeed postings; "
        "predicted negativity uses out-of-fold model scores, so no review is scored by a model trained on it):")
    log(f"1. External validation: model-predicted negativity vs. Indeed company rating: "
        f"{spear(usable, 'predicted_neg_share', 'indeed_rating')}")
    log(f"2. Cross-platform agreement: Glassdoor mean rating vs. Indeed company rating: "
        f"{spear(usable, 'glassdoor_mean_rating', 'indeed_rating')}")
    log(f"3. Salary disclosure indicator: predicted negativity vs. share of postings disclosing salary: "
        f"{spear(usable, 'predicted_neg_share', 'salary_disclosed')}")
    if len(usable):
        log()
        log("Matched companies with the highest predicted negativity:")
        log(table(usable.sort_values("predicted_neg_share", ascending=False).head(10)))

    # ================================================================ LIMITATIONS
    log()
    log("## Limitations and bias")
    log("- Selection bias: people who post reviews skew very happy or very unhappy, so these rates don't "
        "represent all California workers.")
    log("- Star ratings are a proxy for a bad workplace; a 2-star review may be about pay or commute, not mistreatment.")
    log("- Geography mismatch: Glassdoor covers all of California (2025+), Indeed only the Bay Area (2024+). "
        "The join is at the company level, so it mixes locations.")
    log("- Name matching: exact match after normalization misses variants and can merge parent/subsidiary "
        "companies; matches may be incorrect.")
    log("- Subgroups with fewer reviews (see tables) have noisier metrics. Workers in under-represented roles "
        "or regions get less reliable predictions, which matters if a tool like this is used to judge employers.")
    log("- Text models can misread dialect, sarcasm or non-native English; see failures.csv.")
    log("- Ethics: results are reported only at the company or group level; failure examples omit company, "
        "title and location but quotations may still identify reviewers. Keep failures.csv local.")
    log()
    log("## Reproducibility")
    log("Run `python -m workplace_reviews` from the repository root after `pip install -e .`. "
        "Split seed = 42. See README.md and results/VALIDATION.md for environment and run details.")
    report.write(output_dir / "RESULTS.md")
    print(f"Wrote RESULTS.md, failures.csv, company_join.csv to {output_dir}")
