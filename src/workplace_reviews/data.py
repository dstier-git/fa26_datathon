"""Input loading and flexible column discovery."""
import glob
import os
import re
import pandas as pd

def read_any(f):
    ext = f.lower().rsplit(".", 1)[-1]
    if ext == "csv":
        try:
            return pd.read_csv(f, low_memory=False)
        except UnicodeDecodeError:
            return pd.read_csv(f, low_memory=False, encoding="latin-1")
    if ext in ("xlsx", "xls"):
        return pd.read_excel(f)
    if ext == "jsonl":
        return pd.read_json(f, lines=True)
    if ext == "json":
        try:
            return pd.read_json(f)
        except ValueError:
            return pd.read_json(f, lines=True)
    if ext == "parquet":
        return pd.read_parquet(f)
    return None


def load(d):
    exts = ("csv", "xlsx", "xls", "json", "jsonl", "parquet")
    files = [f for f in glob.glob(os.path.join(d, "**", "*"), recursive=True)
             if f.lower().rsplit(".", 1)[-1] in exts]
    full = [f for f in files if "sample" not in os.path.basename(f).lower()]
    use = full or files
    if not use:
        raise ValueError(f"No data files found in {d!r}")
    print(f"[{d}] using: {[os.path.basename(f) for f in use]}")
    return pd.concat([read_any(f) for f in use], ignore_index=True)


def find(df, *cands, required=True):
    keys = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in df.columns}
    cn = [re.sub(r"[^a-z0-9]", "", c.lower()) for c in cands]
    for k in cn:
        if k in keys:
            return keys[k]
    for k in cn:
        for nk, c in keys.items():
            if k in nk:
                return c
    if required:
        raise ValueError(f"Could not find a column like {cands}.\nColumns are: {list(df.columns)}")
    return None
