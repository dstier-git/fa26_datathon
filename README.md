# Workplace Signals

**Can review text predict low workplace ratings at unseen employers—and do those signals agree across platforms?**

We classify Glassdoor reviews using their pros and cons, then connect company-level signals to Indeed ratings and salary disclosure. This is a lightweight, interpretable hackathon experiment, with company-separated evaluation and explicit baselines.

## Quick start

Use Python 3.11 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
# Place the supplied datasets as described in data/README.md
python -m workplace_reviews
```

To use other locations (paths resolve from your current directory):

```bash
python -m workplace_reviews \
  --glassdoor-dir "data/glassdoor reviews dataset" \
  --indeed-dir "data/indeed job listings dataset" \
  --output-dir outputs
```

The full run fits four text models and reads a roughly 2 GB Indeed CSV. Allow several minutes and several GB of available memory. The command overwrites outputs in the selected directory.

## Method

1. Keep reviews with a valid 1–5 rating and nonempty pros or cons. Label ratings ≤2 as negative; ratings 3–5 are the other class.
2. Concatenate pros and cons. Fit TF-IDF unigrams/bigrams (`min_df=3`, up to 200,000 features, sublinear term frequency) and balanced logistic regression (`C=1`, liblinear, up to 2,000 iterations).
3. Hold out 20% of normalized company groups using split seed 42. Compare against majority-class and negative-keyword baselines. Report macro-F1, negative-class precision/recall, false-positive rate, and ROC-AUC; audit status, region, and role groups with at least 50 test reviews.
4. Generate three-fold company-grouped out-of-fold probabilities for every review, independently of the held-out evaluation. Aggregate their mean by company. `predicted_neg_share` is a mean probability, not a fraction of thresholded predictions.
5. Normalize company names using the original punctuation/legal-word rules, join exact normalized names, and deduplicate Indeed postings by title, normalized company, and location where available. Compare company indicators using Spearman correlations for companies with ≥20 reviews and ≥3 postings.

The 80/20 split refers to company groups, not an exact review-count ratio. Vectorizers fit only on their corresponding training folds. Hyperparameters and the classification threshold (0.5) are fixed. The split seed is fixed; the original solver has no explicit random seed, so tiny numerical differences can occur.

## Results

See [the full report](results/RESULTS.md) and [validation details](results/VALIDATION.md).

| Held-out model | Macro-F1 | Negative precision | Negative recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Majority class | 0.443 | 0.000 | 0.000 | — |
| Keyword rule | 0.649 | 0.700 | 0.280 | — |
| TF-IDF + logistic regression | 0.791 | 0.596 | 0.797 | 0.909 |

Verified rerun: 175,138 reviews; 30,555 held-out reviews from 5,245 companies. The cross-platform join matched 5,131 companies, with 633 meeting analysis thresholds. Predicted negativity correlated with Indeed rating at −0.587; salary disclosure correlation was 0.023. Consult validation details for the rerun comparison.

## Outputs and repository

- `src/workplace_reviews/`: loading, preprocessing, modeling, evaluation, reporting, and the pipeline/CLI.
- `tests/`: synthetic checks; run `pytest` after installing the development extra.
- `data/README.md`: input setup; raw datasets stay local.
- `results/`: curated report, aggregate `company_join.csv`, and run validation/environment details.
- `outputs/`: ignored generated `RESULTS.md`, `company_join.csv`, and `failures.csv`. The latter contains high-confidence positive/negative error candidates and review quotations; keep it local. The generated Markdown omits quotations.

The original experiment and separate preprocessing/profiling/research files remain locally available but are ignored and are not dependencies of the submission.

## Interpretation and limitations

Ratings are a proxy for negative experience, not proof of mistreatment. Reviewers self-select, so the sample does not represent all workers. The supplied experiment describes Glassdoor as California-wide (2025+) and Indeed as Bay Area (2024+); the company join mixes locations and periods.

Name normalization can merge unrelated companies and miss aliases. Role and region assignments are substring/regex heuristics, not verified demographics; the default “Other CA” does not validate residency. Employee status mapping may be uninformative (all Unknown in the historical run). Subgroup checks do not establish fairness, and small groups have noisy estimates.

Correlations are observational. Salary disclosure measures nonempty posted salary fields, not pay quality or legal compliance. Review text may contain identifying details even after company/title/location fields are removed. Treat scores as exploratory signals, not employer rankings or employment decisions. Data redistribution rights are not assumed; obtain the organizer-provided datasets separately.
