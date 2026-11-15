# Results: Predicting Negative Workplace Reviews + Glassdoor x Indeed Company Join

## Column mapping
Glassdoor: company=company_name, pros=review_pros, cons=review_cons, rating=rating_overall, status=employee_status, location=employee_location, title=employee_job_title

## Data decisions
- Glassdoor rows loaded: 175,138; kept 175,138 with a valid 1-5 rating and pros/cons text.
- Target: negative review = overall rating <= 2 (positive class rate 19.2%). Labels come from reviewers' own star ratings, so no labels were invented.
- Model input: pros + cons text only (rating never used as a feature).
- Company names normalized (lowercase, punctuation and legal suffixes like Inc/LLC/Corp removed) for grouping and for the Indeed join.
- Train/test split is GROUPED BY COMPANY (80/20): no company appears in both, so the model can't memorize company-specific wording. The test set was never used for training or tuning (hyperparameters are fixed defaults).

## Model vs. baselines (held-out test set: 30,555 reviews from 5,245 companies never seen in training)
```
                        macro_F1  precision_neg  recall_neg  false_pos_rate    AUC
Majority class             0.443          0.000       0.000           0.000    NaN
Keyword rule (cons)        0.649          0.700       0.280           0.031    NaN
TF-IDF + LogReg (ours)     0.791          0.596       0.797           0.139  0.909
```

Top words pushing toward NEGATIVE: toxic, no, pros none, terrible, horrible, poor, worst, zero, pros some, pros nothing, unprofessional, management, awful, favoritism, not, no pros, leadership, incompetent, sometimes cons, pros can

Top words pushing toward POSITIVE: great, no cons, cons none, cons some, cons nothing, good, cons sometimes, amazing, sometimes, bit, supportive, management cons, flexible, excellent, awesome, pros culture, love, can, pros everything, great culture

## Error and fairness analysis (test set, groups with >= 50 reviews)
### By status
```
               n  neg_rate  macro_F1  precision_neg  recall_neg  false_pos_rate    AUC
Unknown  30555.0     0.205     0.791          0.596       0.797           0.139  0.909
```
### By region
```
                      n  neg_rate  macro_F1  precision_neg  recall_neg  false_pos_rate    AUC
Other CA        17986.0     0.194     0.785          0.583       0.781           0.135  0.905
Bay Area         6592.0     0.203     0.789          0.592       0.792           0.139  0.908
LA area          3840.0     0.239     0.804          0.628       0.835           0.155  0.914
San Diego area   1783.0     0.235     0.819          0.647       0.854           0.143  0.927
Sacramento        354.0     0.260     0.796          0.617       0.859           0.187  0.925
```
### By role
```
                           n  neg_rate  macro_F1  precision_neg  recall_neg  false_pos_rate    AUC
Other                10945.0     0.204     0.783          0.583       0.789           0.145  0.906
Tech/Engineering      7581.0     0.167     0.786          0.577       0.758           0.111  0.898
Management            4775.0     0.201     0.789          0.585       0.806           0.144  0.909
Retail/Sales          3409.0     0.179     0.771          0.544       0.782           0.143  0.910
Customer service      1186.0     0.260     0.800          0.635       0.831           0.167  0.915
Unknown               1039.0     0.452     0.884          0.847       0.909           0.135  0.951
Healthcare             651.0     0.237     0.771          0.581       0.792           0.177  0.900
Warehouse/Logistics    369.0     0.295     0.776          0.606       0.862           0.235  0.885
Food service           321.0     0.221     0.762          0.534       0.873           0.216  0.900
Education              279.0     0.276     0.733          0.567       0.714           0.208  0.855
```

## Where the model fails (full examples in failures.csv)
Review quotations are kept only in the local failures.csv; removing identifiers does not guarantee anonymity.

## Dataset join: Glassdoor x Indeed (by normalized company name)
Indeed columns: company=company_name, salary=salary_formatted, company_rating=company_rating
- Indeed postings: 140,816 raw, 136,022 after deduping on title+company+location.
- Companies matched across datasets: 5,131; with >=20 reviews and >=3 postings: 633.
- Matched companies cover 58.7% of Glassdoor reviews and 44.1% of deduped Indeed postings.

Company-level results (companies with >=20 Glassdoor reviews and >=3 Indeed postings; predicted negativity uses out-of-fold model scores, so no review is scored by a model trained on it):
1. External validation: model-predicted negativity vs. Indeed company rating: Spearman rho = -0.587 (n = 628 companies)
2. Cross-platform agreement: Glassdoor mean rating vs. Indeed company rating: Spearman rho = 0.593 (n = 628 companies)
3. Salary disclosure indicator: predicted negativity vs. share of postings disclosing salary: Spearman rho = 0.023 (n = 633 companies)

Matched companies with the highest predicted negativity:
```
                            reviews  glassdoor_mean_rating  actual_neg_share  predicted_neg_share  postings  salary_disclosed  indeed_rating
co
veritone                         31                  2.000             0.710                0.766         9             1.000          2.500
public storage                   31                  2.452             0.516                0.680        11             0.909          3.200
weedmaps                         44                  2.068             0.705                0.674         6             1.000          2.900
axos bank                       156                  2.167             0.667                0.670        45             0.978          3.100
scan health plan                 47                  2.255             0.638                0.666        18             1.000          3.700
lifestance health                36                  2.278             0.639                0.655        28             0.964          3.096
hd supply                        21                  2.524             0.571                0.610        10             0.900          3.300
ssp                              26                  2.577             0.500                0.604         7             1.000          2.800
lucky strike entertainment       23                  2.522             0.478                0.595         5             1.000          3.000
postman                          39                  2.667             0.538                0.587         3             0.333            NaN
```

## Limitations and bias
- Selection bias: people who post reviews skew very happy or very unhappy, so these rates don't represent all California workers.
- Star ratings are a proxy for a bad workplace; a 2-star review may be about pay or commute, not mistreatment.
- Geography mismatch: Glassdoor covers all of California (2025+), Indeed only the Bay Area (2024+). The join is at the company level, so it mixes locations.
- Name matching: exact match after normalization misses variants and can merge parent/subsidiary companies; matches may be incorrect.
- Subgroups with fewer reviews (see tables) have noisier metrics. Workers in under-represented roles or regions get less reliable predictions, which matters if a tool like this is used to judge employers.
- Text models can misread dialect, sarcasm or non-native English; see failures.csv.
- Ethics: results are reported only at the company or group level; failure examples omit company, title and location but quotations may still identify reviewers. Keep failures.csv local.

## Reproducibility
Run `python -m workplace_reviews` from the repository root after `pip install -e .`. Split seed = 42. See README.md and results/VALIDATION.md for environment and run details.
