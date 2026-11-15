"""Original Indeed deduplication and exact normalized-company join."""
import numpy as np
import pandas as pd
from .preprocessing import norm_co


def join_companies(gd, ind, I_CO, I_SAL, I_RAT, I_TITLE, I_LOC):
    ind["co"] = ind[I_CO].map(norm_co)
    n_ind_raw = len(ind)
    dedupe_cols = [c for c in [I_TITLE, "co", I_LOC] if c]
    ind = ind.drop_duplicates(subset=dedupe_cols)
    if I_SAL:
        s = ind[I_SAL].astype(str).str.strip().str.lower()
        ind["has_salary"] = (~s.isin(["", "nan", "none", "null", "n/a"])).astype(float)
    else:
        ind["has_salary"] = np.nan
    ind["co_rating"] = pd.to_numeric(ind[I_RAT], errors="coerce") if I_RAT else np.nan
    ind = ind[ind["co"] != "unknown"]

    ind_co = ind.groupby("co").agg(postings=("co", "size"),
                                   salary_disclosed=("has_salary", "mean"),
                                   indeed_rating=("co_rating", "mean"))
    gd_co = gd.groupby("co").agg(reviews=("y", "size"),
                                 glassdoor_mean_rating=("rating_num", "mean"),
                                 actual_neg_share=("y", "mean"),
                                 predicted_neg_share=("oof_prob", "mean"))
    joined = gd_co.join(ind_co, how="inner")
    usable = joined[(joined["reviews"] >= 20) & (joined["postings"] >= 3)]

    return ind, joined, usable, n_ind_raw
