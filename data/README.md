# Input datasets

Obtain the original Glassdoor Reviews and Indeed Job Listings datasets from the datathon organizers. Raw files and organizer documents are intentionally excluded from Git.

Place the full files here:

```text
data/
  glassdoor reviews dataset/glassdoor_reviews.csv
  indeed job listings dataset/indeed_job_listings.csv
```

The original files are approximately 132 MB and 2 GB. Keep unrelated tables outside these folders: the loader recursively concatenates supported files. Files with “sample” in their basename are excluded if any nonsample data files exist; otherwise sample files are used.

Supported formats: CSV (UTF-8, with Latin-1 fallback), JSON, JSONL, XLS/XLSX, and Parquet. Install `pip install -e '.[formats]'` for Excel/Parquet support.

Glassdoor requires company, pros, cons, and overall rating columns. The supplied mapping is `company_name`, `review_pros`, `review_cons`, `rating_overall`; optional subgroup fields are `employee_status`, `employee_location`, and `employee_job_title`. Indeed requires company (`company_name`); optional fields are `salary_formatted`, `company_rating`, `job_title`, and `location`. The loader accepts normalized column aliases and reports the resolved mapping. Missing optional fields yield unknown subgroups or missing indicators.

Full evaluation needs enough distinct companies and both rating classes in training folds. Tiny samples may fail the fixed three-fold split or TF-IDF minimum document frequency; use the synthetic test fixture for a smoke check.
