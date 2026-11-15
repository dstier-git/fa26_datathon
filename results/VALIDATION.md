# Validation

Verified on 2026-11-15 using Python 3.14.6 on macOS. Exact runtime and test dependency versions are in [environment.txt](environment.txt). The original experiment did not record its dependency versions.

## Reproduce the verified environment

From the repository root with Python 3.14.6:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r results/environment.txt
pip install -e .
pytest -q
python -m workplace_reviews
```

Other supported Python/dependency versions may produce different fold assignments or numerical results. The original liblinear model does not explicitly seed the solver; this refactor preserves that configuration. The company holdout split uses seed 42.

## Checks completed

- Four synthetic tests passed in both the working environment and a separate fresh virtual environment installed from the recorded dependencies and project metadata.
- Tests cover import/help without data access, sample-file exclusion, company isolation in the holdout and all three out-of-fold splits, exactly one out-of-fold score per review, rating-derived labels through aggregate counts, and end-to-end output paths.
- Missing-input CLI check returned a readable error.
- Full-data CLI execution completed and generated all three expected artifacts.
- The untouched original `experiment1/run_all.py` was independently run on the same full datasets in the same environment, writing to a temporary directory. All company CSV keys and numeric values matched the packaged run exactly, including predicted probabilities. Held-out model metrics and correlations matched at displayed precision.
- Git inventory contains the package, tests, documentation, curated results, and environment record. Raw datasets, quoted failures, original experiments, local tools, and environments are ignored. No commits or pushes were made during validation.

## Comparison with the historical saved experiment

| Measure | Historical saved report | Verified packaged run |
|---|---:|---:|
| Glassdoor reviews retained | 175,138 | 175,138 |
| Held-out reviews / companies | 30,555 / 5,245 | 30,555 / 5,245 |
| Macro-F1 / ROC-AUC | 0.791 / 0.909 | 0.791 / 0.909 |
| Indeed raw / deduplicated postings | 140,816 / 136,022 | 140,816 / 136,022 |
| Joined / threshold-qualified companies | 5,131 / 633 | 5,131 / 633 |
| Predicted negativity vs. Indeed rating | −0.583 | −0.587 |
| Glassdoor vs. Indeed rating | 0.593 | 0.593 |
| Predicted negativity vs. salary disclosure | 0.023 | 0.023 |

Historical aggregate CSV keys and all nonprediction fields matched exactly. Historical out-of-fold company probabilities differed (maximum absolute company-level difference 0.264354, including small companies). The historical run's exact environment and provenance are unavailable, so the cause of that difference is unconfirmed. Running the untouched source under the recorded current environment reproduced the new values exactly; no change to the method is needed to produce them.

The curated report and CSV are from the verified packaged run. Reports omit review quotations; `failures.csv` remains local. These checks validate execution and preservation of the implemented experiment, not the validity of its proxy labels or causal interpretation.
