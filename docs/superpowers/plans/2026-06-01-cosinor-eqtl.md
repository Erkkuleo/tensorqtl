# Cosinor eQTL Mapping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement circadian/cosinor eQTL mapping tools that run a 1-DF interaction test (SNP × cos_t) using tensorQTL's `cis.map_nominal` API.

**Architecture:** Two independent scripts: `cosinor_preprocess.py` computes cosinor covariates and the interaction file from sample metadata; `run_cosinor_qtl.py` loads all inputs and calls tensorQTL. Separation allows each script to be used independently or swapped. A single `parse_time_to_hours` stub isolates all time-parsing logic.

**Tech Stack:** Python 3.11, tensorQTL (Python API), pandas, numpy, scipy, pytest.

---

## File Structure

```
tensorqtl/                            ← repo root (already exists)
├── scripts/
│   ├── cosinor_preprocess.py         ← NEW: compute cos_t/sin_t, write covariates + interaction files
│   └── run_cosinor_qtl.py            ← NEW: load inputs, call tensorQTL cis_nominal with interaction
├── tests/
│   ├── test_cosinor_preprocess.py    ← NEW: unit tests for all preprocessing functions
│   └── test_run_cosinor_qtl.py       ← NEW: unit tests for validate_sample_alignment
└── docs/
    └── cosinor_eqtl.md               ← NEW: scientific rationale, upgrade paths
```

## Data Flow

```
metadata.tsv  ──────────────────┐
                                 ├──► cosinor_preprocess.py ──► covariates_with_cosinor.txt
covariates.txt ─────────────────┘                           ──► interaction_cos_t.txt
                                                                       │
plink_prefix (pgen/psam/pvar) ─────────────────────────────────────────┤
phenotypes.bed.gz ──────────────────────────────────────────────────────┤
covariates_with_cosinor.txt ────────────────────────────────────────────┤
interaction_cos_t.txt ──────────────────────────────────────────────────┘
         │
         ▼
   run_cosinor_qtl.py
         │
         ▼
   <prefix>.cis_qtl_pairs.<chr>.parquet  (one per chromosome)
   <prefix>.cis_qtl_top_assoc.txt.gz
```

## File Format Notes

- **Input covariates** (tensorQTL standard): TSV, rows = covariate names, columns = sample IDs. First column is covariate name; first row is sample IDs (with a tab/empty field in position 0,0).
- **Output covariates**: Same format as input, with `cos_t` and `sin_t` rows appended.
- **Interaction file**: TSV, first column = sample IDs (index), column header = `cos_t`. Written by `interaction_df.to_csv(..., sep="\t")` and read by `pd.read_csv(..., index_col=0)`.
- **Metadata file**: TSV, first column = sample IDs (used as index), must contain the `--time-col` column.

## Function Signatures

### `scripts/cosinor_preprocess.py`

```python
def parse_time_to_hours(value: str) -> float
    # STUB: float(value). Isolated upgrade point for ISO 8601 parsing.

def compute_cosinor(hours: pd.Series, period: float = 24.0) -> tuple[pd.Series, pd.Series]
    # Returns (cos_t, sin_t), both Series indexed by sample ID.

def load_metadata(path: str, time_col: str) -> pd.Series
    # TSV with sample IDs in first column. Returns hours Series.
    # Raises ValueError on missing time_col, duplicate sample IDs.

def load_covariates(path: str) -> pd.DataFrame
    # rows=covariates, cols=samples. Returns as-is.

def append_cosinor_to_covariates(covariates_df, cos_t, sin_t) -> pd.DataFrame
    # Appends cos_t and sin_t rows. Raises ValueError if already present.

def make_interaction_df(cos_t: pd.Series) -> pd.DataFrame
    # Returns DataFrame(index=sample_IDs, columns=["cos_t"]).

def main() -> None
    # CLI entry point.
```

### `scripts/run_cosinor_qtl.py`

```python
def load_inputs(plink_prefix, phenotypes_path, covariates_path, interaction_path) -> tuple
    # Returns (genotype_df, variant_df, phenotype_df, phenotype_pos_df,
    #          covariates_df, interaction_df)
    # covariates_df: samples × covariates (transposed from file)
    # interaction_df: samples × 1 (cos_t), indexed by sample ID

def validate_sample_alignment(phenotype_df, covariates_df, interaction_df) -> None
    # Raises ValueError with informative message if samples mismatch or wrong order.

def run_cosinor_mapping(genotype_df, variant_df, phenotype_df, phenotype_pos_df,
                        covariates_df, interaction_df, prefix, output_dir,
                        window=1_000_000) -> None
    # Calls cis.map_nominal with interaction_df. Creates output_dir if needed.

def main() -> None
    # CLI entry point.
```

## Upgrade Paths

### ISO 8601 timestamps
Change only `parse_time_to_hours` in `cosinor_preprocess.py`:
```python
# Current stub:
def parse_time_to_hours(value: str) -> float:
    return float(value)

# Future replacement:
from datetime import datetime
def parse_time_to_hours(value: str) -> float:
    dt = datetime.fromisoformat(value)
    return dt.hour + dt.minute / 60.0 + dt.second / 3600.0
```
No other code changes needed — all downstream logic consumes the returned float.

### 2-DF joint test (SNP×cos_t and SNP×sin_t)
In `cosinor_preprocess.py`: change `make_interaction_df` to return a 2-column DataFrame:
```python
# Current (1-DF):
return cos_t.to_frame(name="cos_t")

# Future (2-DF):
return pd.DataFrame({"cos_t": cos_t, "sin_t": sin_t})
```
`cis.map_nominal` already handles multiple interaction columns — it reports individual t-statistics per interaction term. A combined 2-DF F-test requires additional post-processing on the output (combining the two t-stats into a chi-squared statistic with 2 DF), which is not yet implemented.

---

## Task 1: Project skeleton and test infrastructure

**Files:**
- Create: `scripts/__init__.py` (empty)
- Create: `tests/__init__.py` (empty)
- Create: `tests/test_cosinor_preprocess.py` (imports only, verifies import works)
- Create: `tests/test_run_cosinor_qtl.py` (imports only, verifies import works)

- [ ] **Step 1: Create the directory structure**

```bash
mkdir -p scripts tests
touch scripts/__init__.py tests/__init__.py
```

- [ ] **Step 2: Create minimal test stubs to confirm pytest runs**

Create `tests/test_cosinor_preprocess.py`:
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))


def test_placeholder():
    pass
```

Create `tests/test_run_cosinor_qtl.py`:
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))


def test_placeholder():
    pass
```

- [ ] **Step 3: Run tests to confirm infrastructure works**

```bash
pytest tests/ -v
```

Expected output:
```
tests/test_cosinor_preprocess.py::test_placeholder PASSED
tests/test_run_cosinor_qtl.py::test_placeholder PASSED
2 passed in 0.XXs
```

- [ ] **Step 4: Commit**

```bash
git add scripts/__init__.py tests/__init__.py tests/test_cosinor_preprocess.py tests/test_run_cosinor_qtl.py
git commit -m "chore: add scripts/ and tests/ skeleton for cosinor eQTL tools"
```

---

## Task 2: `parse_time_to_hours` and `compute_cosinor`

**Files:**
- Create: `scripts/cosinor_preprocess.py`
- Modify: `tests/test_cosinor_preprocess.py`

- [ ] **Step 1: Write failing tests for `parse_time_to_hours` and `compute_cosinor`**

Replace `tests/test_cosinor_preprocess.py` with:
```python
import pytest
import math
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from cosinor_preprocess import parse_time_to_hours, compute_cosinor


def test_parse_time_to_hours_numeric_string():
    assert parse_time_to_hours("12.5") == 12.5


def test_parse_time_to_hours_integer_string():
    assert parse_time_to_hours("0") == 0.0


def test_parse_time_to_hours_nonnumeric_raises():
    with pytest.raises(ValueError):
        parse_time_to_hours("not_a_number")


def test_compute_cosinor_known_values():
    hours = pd.Series({"S1": 0.0, "S2": 6.0, "S3": 12.0, "S4": 18.0})
    cos_t, sin_t = compute_cosinor(hours, period=24.0)
    assert abs(cos_t["S1"] - 1.0) < 1e-6    # cos(0) = 1
    assert abs(cos_t["S2"] - 0.0) < 1e-6    # cos(π/2) = 0
    assert abs(cos_t["S3"] - (-1.0)) < 1e-6  # cos(π) = -1
    assert abs(cos_t["S4"] - 0.0) < 1e-6    # cos(3π/2) ≈ 0
    assert abs(sin_t["S1"] - 0.0) < 1e-6    # sin(0) = 0
    assert abs(sin_t["S2"] - 1.0) < 1e-6    # sin(π/2) = 1
    assert abs(sin_t["S3"]) < 1e-6          # sin(π) ≈ 0 (floating point)
    assert abs(sin_t["S4"] - (-1.0)) < 1e-6  # sin(3π/2) = -1


def test_compute_cosinor_preserves_index():
    hours = pd.Series({"A": 3.0, "B": 9.0})
    cos_t, sin_t = compute_cosinor(hours)
    assert list(cos_t.index) == ["A", "B"]
    assert list(sin_t.index) == ["A", "B"]


def test_compute_cosinor_series_names():
    hours = pd.Series({"S1": 6.0})
    cos_t, sin_t = compute_cosinor(hours)
    assert cos_t.name == "cos_t"
    assert sin_t.name == "sin_t"


def test_compute_cosinor_custom_period():
    # 12-hour period: hour=6 → angle=π → cos=-1
    hours = pd.Series({"S1": 6.0})
    cos_t, _ = compute_cosinor(hours, period=12.0)
    assert abs(cos_t["S1"] - (-1.0)) < 1e-6
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_cosinor_preprocess.py -v
```

Expected: `ModuleNotFoundError: No module named 'cosinor_preprocess'`

- [ ] **Step 3: Create `scripts/cosinor_preprocess.py` with the two functions**

```python
#!/usr/bin/env python3
"""Cosinor preprocessing for circadian eQTL mapping."""
import math
import numpy as np
import pandas as pd


def parse_time_to_hours(value: str) -> float:
    """Convert a time value string to fractional hours (0–24).

    STUB: currently casts to float. To support ISO 8601 timestamps
    (e.g., "2024-01-15T14:30:00"), replace the body of this function
    with ISO 8601 parsing. All downstream code consumes the returned
    float, so no other changes are needed.

    Replacement example:
        from datetime import datetime
        dt = datetime.fromisoformat(value)
        return dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        raise ValueError(
            f"Cannot convert '{value}' to hours. Expected a numeric value (e.g. '14.5')."
        )


def compute_cosinor(hours: pd.Series, period: float = 24.0) -> tuple:
    """Compute cosine and sine time encodings for circadian eQTL mapping.

    Args:
        hours: Series indexed by sample ID, values are fractional hours (0–24).
        period: Cycle period in hours (default 24.0 for circadian rhythm).

    Returns:
        Tuple of (cos_t, sin_t), each a named Series indexed by sample ID.
    """
    angle = 2.0 * math.pi * hours / period
    cos_t = pd.Series(np.cos(angle.values), index=hours.index, name="cos_t")
    sin_t = pd.Series(np.sin(angle.values), index=hours.index, name="sin_t")
    return cos_t, sin_t
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_cosinor_preprocess.py -v -k "parse_time or compute_cosinor"
```

Expected: 6 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add scripts/cosinor_preprocess.py tests/test_cosinor_preprocess.py
git commit -m "feat: add parse_time_to_hours stub and compute_cosinor"
```

---

## Task 3: `load_metadata`, `load_covariates`, `append_cosinor_to_covariates`, `make_interaction_df`

**Files:**
- Modify: `scripts/cosinor_preprocess.py`
- Modify: `tests/test_cosinor_preprocess.py`

- [ ] **Step 1: Write failing tests for the four new functions**

Append to `tests/test_cosinor_preprocess.py`:
```python
from cosinor_preprocess import (
    load_metadata, load_covariates,
    append_cosinor_to_covariates, make_interaction_df,
)


def test_load_metadata_basic(tmp_path):
    f = tmp_path / "meta.tsv"
    f.write_text("sample_id\thour\nS1\t8.0\nS2\t20.5\n")
    result = load_metadata(str(f), time_col="hour")
    assert len(result) == 2
    assert result["S1"] == pytest.approx(8.0)
    assert result["S2"] == pytest.approx(20.5)


def test_load_metadata_missing_column_raises(tmp_path):
    f = tmp_path / "meta.tsv"
    f.write_text("sample_id\ttime\nS1\t8.0\n")
    with pytest.raises(ValueError, match="time_col 'hour'"):
        load_metadata(str(f), time_col="hour")


def test_load_metadata_duplicate_sample_raises(tmp_path):
    f = tmp_path / "meta.tsv"
    f.write_text("sample_id\thour\nS1\t8.0\nS1\t12.0\n")
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        load_metadata(str(f), time_col="hour")


def test_load_metadata_nonnumeric_time_raises(tmp_path):
    f = tmp_path / "meta.tsv"
    f.write_text("sample_id\thour\nS1\tnoon\n")
    with pytest.raises(ValueError):
        load_metadata(str(f), time_col="hour")


def test_load_covariates_basic(tmp_path):
    f = tmp_path / "cov.txt"
    f.write_text("ID\tS1\tS2\nPC1\t0.1\t0.2\nPC2\t-0.1\t0.3\n")
    result = load_covariates(str(f))
    assert result.shape == (2, 2)
    assert list(result.index) == ["PC1", "PC2"]
    assert list(result.columns) == ["S1", "S2"]


def test_append_cosinor_to_covariates_adds_rows():
    cov = pd.DataFrame({"S1": [0.1], "S2": [0.2]}, index=["PC1"])
    cos_t = pd.Series({"S1": 0.866, "S2": 0.5})
    sin_t = pd.Series({"S1": 0.5, "S2": 0.866})
    result = append_cosinor_to_covariates(cov, cos_t, sin_t)
    assert list(result.index) == ["PC1", "cos_t", "sin_t"]
    assert result.loc["cos_t", "S1"] == pytest.approx(0.866)
    assert result.loc["sin_t", "S2"] == pytest.approx(0.866)


def test_append_cosinor_raises_if_cos_t_already_present():
    cov = pd.DataFrame({"S1": [0.1]}, index=["cos_t"])
    cos_t = pd.Series({"S1": 0.866})
    sin_t = pd.Series({"S1": 0.5})
    with pytest.raises(ValueError, match="already present"):
        append_cosinor_to_covariates(cov, cos_t, sin_t)


def test_append_cosinor_raises_if_sin_t_already_present():
    cov = pd.DataFrame({"S1": [0.1]}, index=["sin_t"])
    cos_t = pd.Series({"S1": 0.866})
    sin_t = pd.Series({"S1": 0.5})
    with pytest.raises(ValueError, match="already present"):
        append_cosinor_to_covariates(cov, cos_t, sin_t)


def test_make_interaction_df_shape_and_name():
    cos_t = pd.Series({"S1": 0.866, "S2": 0.5})
    result = make_interaction_df(cos_t)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 1)
    assert result.columns[0] == "cos_t"
    assert list(result.index) == ["S1", "S2"]
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_cosinor_preprocess.py -v -k "load_metadata or load_covariates or append_cosinor or make_interaction"
```

Expected: `ImportError` or `FAILED` for each new test.

- [ ] **Step 3: Implement the four functions in `scripts/cosinor_preprocess.py`**

Append to `scripts/cosinor_preprocess.py`:
```python

def load_metadata(path: str, time_col: str) -> pd.Series:
    """Load sample metadata and return hour-of-day values.

    File must be tab-separated with a header; the first column is used as
    sample IDs.

    Args:
        path: Path to TSV metadata file.
        time_col: Column name containing time-of-day values.

    Returns:
        Series of float hours indexed by sample ID.

    Raises:
        ValueError: If time_col is absent, sample IDs are duplicated,
                    or any time value cannot be converted to float.
    """
    df = pd.read_csv(path, sep="\t", index_col=0)
    if time_col not in df.columns:
        raise ValueError(
            f"time_col '{time_col}' not found in metadata. "
            f"Available columns: {list(df.columns)}"
        )
    if df.index.duplicated().any():
        dupes = df.index[df.index.duplicated()].tolist()
        raise ValueError(f"Duplicate sample IDs in metadata: {dupes}")
    return df[time_col].map(lambda v: parse_time_to_hours(str(v)))


def load_covariates(path: str) -> pd.DataFrame:
    """Load a tensorQTL-format covariates file.

    Format: rows = covariate names, columns = sample IDs (tab-separated).

    Returns:
        DataFrame with covariate names as index, sample IDs as columns.
    """
    return pd.read_csv(path, sep="\t", index_col=0)


def append_cosinor_to_covariates(
    covariates_df: pd.DataFrame, cos_t: pd.Series, sin_t: pd.Series
) -> pd.DataFrame:
    """Append cos_t and sin_t rows to a tensorQTL covariates DataFrame.

    Args:
        covariates_df: Existing covariates (rows=covariate names, cols=samples).
        cos_t: Cosine values indexed by sample ID (must cover all samples).
        sin_t: Sine values indexed by sample ID (must cover all samples).

    Returns:
        New DataFrame with cos_t and sin_t appended as the last two rows.

    Raises:
        ValueError: If 'cos_t' or 'sin_t' are already present in the index.
    """
    for name in ("cos_t", "sin_t"):
        if name in covariates_df.index:
            raise ValueError(
                f"'{name}' already present in covariates index. "
                "Remove it before re-running cosinor_preprocess.py."
            )
    new_rows = pd.DataFrame(
        [
            cos_t.reindex(covariates_df.columns).values,
            sin_t.reindex(covariates_df.columns).values,
        ],
        index=["cos_t", "sin_t"],
        columns=covariates_df.columns,
    )
    return pd.concat([covariates_df, new_rows])


def make_interaction_df(cos_t: pd.Series) -> pd.DataFrame:
    """Create a tensorQTL interaction DataFrame for the cos_t term.

    To upgrade to a 2-DF test, change this function to return a
    2-column DataFrame: pd.DataFrame({"cos_t": cos_t, "sin_t": sin_t}).

    Returns:
        DataFrame with sample IDs as index, single column 'cos_t'.
    """
    return cos_t.to_frame(name="cos_t")
```

- [ ] **Step 4: Run all preprocessing tests**

```bash
pytest tests/test_cosinor_preprocess.py -v
```

Expected: all tests PASSED (should be ~15 tests now).

- [ ] **Step 5: Commit**

```bash
git add scripts/cosinor_preprocess.py tests/test_cosinor_preprocess.py
git commit -m "feat: add load/append/interaction functions to cosinor_preprocess"
```

---

## Task 4: `cosinor_preprocess.py` CLI (`main()`) and integration test

**Files:**
- Modify: `scripts/cosinor_preprocess.py`
- Modify: `tests/test_cosinor_preprocess.py`

- [ ] **Step 1: Write integration test for the full CLI flow**

Append to `tests/test_cosinor_preprocess.py`:
```python
import subprocess


def test_main_cli_end_to_end(tmp_path):
    # Build inputs
    meta = tmp_path / "meta.tsv"
    meta.write_text(
        "sample_id\thour\n"
        "S1\t0.0\n"
        "S2\t6.0\n"
        "S3\t12.0\n"
    )
    cov = tmp_path / "cov.txt"
    cov.write_text("ID\tS1\tS2\tS3\nPC1\t0.1\t0.2\t0.3\n")

    out_cov = tmp_path / "out_cov.txt"
    out_int = tmp_path / "out_int.txt"

    result = subprocess.run(
        [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "..", "scripts", "cosinor_preprocess.py"),
            "--metadata", str(meta),
            "--covariates", str(cov),
            "--out-covariates", str(out_cov),
            "--out-interaction", str(out_int),
            "--time-col", "hour",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    # covariates file: 3 rows (PC1 + cos_t + sin_t), 3 columns (S1 S2 S3)
    out_cov_df = pd.read_csv(str(out_cov), sep="\t", index_col=0)
    assert list(out_cov_df.index) == ["PC1", "cos_t", "sin_t"]
    assert abs(out_cov_df.loc["cos_t", "S1"] - 1.0) < 1e-6   # cos(0) = 1
    assert abs(out_cov_df.loc["cos_t", "S2"] - 0.0) < 1e-6   # cos(π/2) = 0
    assert abs(out_cov_df.loc["cos_t", "S3"] - (-1.0)) < 1e-6 # cos(π) = -1

    # interaction file: 3 rows, 1 column (cos_t)
    out_int_df = pd.read_csv(str(out_int), sep="\t", index_col=0)
    assert list(out_int_df.columns) == ["cos_t"]
    assert list(out_int_df.index) == ["S1", "S2", "S3"]


def test_main_cli_missing_samples_raises(tmp_path):
    # Metadata has S1, S2; covariates has S1, S2, S3 → S3 missing from metadata
    meta = tmp_path / "meta.tsv"
    meta.write_text("sample_id\thour\nS1\t0.0\nS2\t6.0\n")
    cov = tmp_path / "cov.txt"
    cov.write_text("ID\tS1\tS2\tS3\nPC1\t0.1\t0.2\t0.3\n")
    out_cov = tmp_path / "out_cov.txt"
    out_int = tmp_path / "out_int.txt"

    result = subprocess.run(
        [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "..", "scripts", "cosinor_preprocess.py"),
            "--metadata", str(meta),
            "--covariates", str(cov),
            "--out-covariates", str(out_cov),
            "--out-interaction", str(out_int),
            "--time-col", "hour",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "S3" in result.stderr or "S3" in result.stdout
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_cosinor_preprocess.py::test_main_cli_end_to_end \
       tests/test_cosinor_preprocess.py::test_main_cli_missing_samples_raises -v
```

Expected: both FAILED (no main() yet).

- [ ] **Step 3: Implement `main()` in `scripts/cosinor_preprocess.py`**

Append to `scripts/cosinor_preprocess.py`:
```python
import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute cosinor (circadian) covariates for tensorQTL interaction mapping. "
            "Appends cos_t and sin_t to an existing covariates file and writes a separate "
            "interaction file containing only cos_t."
        )
    )
    parser.add_argument("--metadata", required=True,
                        help="Sample metadata TSV. First column = sample IDs; must contain --time-col.")
    parser.add_argument("--covariates", required=True,
                        help="Existing tensorQTL covariates file (rows=covariates, cols=samples).")
    parser.add_argument("--out-covariates", required=True,
                        help="Output path for updated covariates (existing rows + cos_t + sin_t).")
    parser.add_argument("--out-interaction", required=True,
                        help="Output path for interaction file (samples × 1 column: cos_t).")
    parser.add_argument("--time-col", default="hour",
                        help="Column name for time-of-day in metadata (default: 'hour').")
    parser.add_argument("--period", type=float, default=24.0,
                        help="Cycle period in hours (default: 24.0 for circadian).")
    args = parser.parse_args()

    print(f"Loading metadata from {args.metadata}")
    hours = load_metadata(args.metadata, time_col=args.time_col)

    print(f"Loading covariates from {args.covariates}")
    covariates_df = load_covariates(args.covariates)

    covariate_samples = pd.Index(covariates_df.columns)
    missing = covariate_samples.difference(hours.index)
    if len(missing) > 0:
        print(
            f"ERROR: {len(missing)} samples in covariates have no metadata entry: "
            f"{missing.tolist()[:10]}",
            file=sys.stderr,
        )
        sys.exit(1)

    hours_aligned = hours.loc[covariate_samples]
    cos_t, sin_t = compute_cosinor(hours_aligned, period=args.period)

    updated_cov = append_cosinor_to_covariates(covariates_df, cos_t, sin_t)
    updated_cov.to_csv(args.out_covariates, sep="\t")
    print(f"Wrote updated covariates ({updated_cov.shape[0]} rows) to {args.out_covariates}")

    interaction_df = make_interaction_df(cos_t)
    interaction_df.to_csv(args.out_interaction, sep="\t")
    print(f"Wrote interaction file ({interaction_df.shape[0]} samples) to {args.out_interaction}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run all tests**

```bash
pytest tests/test_cosinor_preprocess.py -v
```

Expected: all tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add scripts/cosinor_preprocess.py tests/test_cosinor_preprocess.py
git commit -m "feat: add cosinor_preprocess.py CLI with main() and integration tests"
```

---

## Task 5: `validate_sample_alignment` in `run_cosinor_qtl.py`

**Files:**
- Create: `scripts/run_cosinor_qtl.py`
- Modify: `tests/test_run_cosinor_qtl.py`

- [ ] **Step 1: Write failing tests for `validate_sample_alignment`**

Replace `tests/test_run_cosinor_qtl.py`:
```python
import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from run_cosinor_qtl import validate_sample_alignment


def _make_inputs(pheno_samples, cov_samples, int_samples):
    phenotype_df = pd.DataFrame(
        np.zeros((2, len(pheno_samples))), columns=pheno_samples
    )
    covariates_df = pd.DataFrame(
        np.zeros((len(cov_samples), 2)), index=cov_samples
    )
    interaction_df = pd.DataFrame(
        np.zeros((len(int_samples), 1)), index=int_samples, columns=["cos_t"]
    )
    return phenotype_df, covariates_df, interaction_df


def test_validate_ok():
    pheno, cov, inter = _make_inputs(["S1", "S2", "S3"],
                                     ["S1", "S2", "S3"],
                                     ["S1", "S2", "S3"])
    validate_sample_alignment(pheno, cov, inter)  # must not raise


def test_validate_covariate_samples_missing():
    pheno, cov, inter = _make_inputs(["S1", "S2", "S3"],
                                     ["S1", "S2"],
                                     ["S1", "S2", "S3"])
    with pytest.raises(ValueError, match="covariate"):
        validate_sample_alignment(pheno, cov, inter)


def test_validate_covariate_samples_wrong_order():
    pheno, cov, inter = _make_inputs(["S1", "S2", "S3"],
                                     ["S3", "S1", "S2"],
                                     ["S1", "S2", "S3"])
    with pytest.raises(ValueError, match="order"):
        validate_sample_alignment(pheno, cov, inter)


def test_validate_interaction_samples_missing():
    pheno, cov, inter = _make_inputs(["S1", "S2", "S3"],
                                     ["S1", "S2", "S3"],
                                     ["S1", "S2"])
    with pytest.raises(ValueError, match="interaction"):
        validate_sample_alignment(pheno, cov, inter)


def test_validate_interaction_samples_wrong_order():
    pheno, cov, inter = _make_inputs(["S1", "S2", "S3"],
                                     ["S1", "S2", "S3"],
                                     ["S2", "S1", "S3"])
    with pytest.raises(ValueError, match="order"):
        validate_sample_alignment(pheno, cov, inter)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_run_cosinor_qtl.py -v
```

Expected: `ModuleNotFoundError: No module named 'run_cosinor_qtl'`

- [ ] **Step 3: Create `scripts/run_cosinor_qtl.py` with `validate_sample_alignment`**

```python
#!/usr/bin/env python3
"""tensorQTL runner for cosinor (circadian) cis-eQTL interaction mapping.

Runs a 1-DF interaction test: does a SNP's effect on gene expression
change with the cosine of time-of-day?

Model fitted per gene:
    expression ~ SNP + cos_t + sin_t + SNP×cos_t + other_covariates

cos_t and sin_t enter as regular covariates (via --covariates from
cosinor_preprocess.py). cos_t additionally enters as the interaction
term (via --interaction). This is a 1-DF test; for the 2-DF upgrade
see docs/cosinor_eqtl.md.
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd


def validate_sample_alignment(
    phenotype_df: pd.DataFrame,
    covariates_df: pd.DataFrame,
    interaction_df: pd.DataFrame,
) -> None:
    """Verify that all three inputs share the same samples in the same order.

    Args:
        phenotype_df: genes × samples (samples are columns)
        covariates_df: samples × covariates (samples are index)
        interaction_df: samples × interactions (samples are index)

    Raises:
        ValueError: If samples are mismatched or in a different order.
    """
    pheno_samples = phenotype_df.columns

    # Check covariates
    if not pheno_samples.equals(covariates_df.index):
        if set(pheno_samples) == set(covariates_df.index):
            raise ValueError(
                "covariate samples match phenotype samples but are in wrong order. "
                "Reorder the covariates file to match phenotype column order."
            )
        missing = pheno_samples.difference(covariates_df.index).tolist()
        extra = covariates_df.index.difference(pheno_samples).tolist()
        raise ValueError(
            f"covariate samples don't match phenotype samples. "
            f"Missing from covariates: {missing[:5]}. "
            f"Extra in covariates: {extra[:5]}."
        )

    # Check interaction
    if not pheno_samples.equals(interaction_df.index):
        if set(pheno_samples) == set(interaction_df.index):
            raise ValueError(
                "interaction samples match phenotype samples but are in wrong order. "
                "Reorder the interaction file to match phenotype column order."
            )
        missing = pheno_samples.difference(interaction_df.index).tolist()
        extra = interaction_df.index.difference(pheno_samples).tolist()
        raise ValueError(
            f"interaction samples don't match phenotype samples. "
            f"Missing from interaction: {missing[:5]}. "
            f"Extra in interaction: {extra[:5]}."
        )
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_run_cosinor_qtl.py -v
```

Expected: 5 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_cosinor_qtl.py tests/test_run_cosinor_qtl.py
git commit -m "feat: add run_cosinor_qtl.py with validate_sample_alignment"
```

---

## Task 6: `load_inputs`, `run_cosinor_mapping`, and `main()` in `run_cosinor_qtl.py`

**Files:**
- Modify: `scripts/run_cosinor_qtl.py`

These functions depend on tensorQTL and PLINK files, so they are not unit-tested here. They are validated end-to-end in Task 7 using the example data.

- [ ] **Step 1: Append `load_inputs`, `run_cosinor_mapping`, and `main()` to `scripts/run_cosinor_qtl.py`**

```python

def load_inputs(
    plink_prefix: str,
    phenotypes_path: str,
    covariates_path: str,
    interaction_path: str,
) -> tuple:
    """Load all inputs for cosinor cis-QTL mapping.

    Args:
        plink_prefix: Prefix for PLINK2 pgen/psam/pvar files (no extension).
        phenotypes_path: Expression BED file (.bed.gz or .bed.parquet).
        covariates_path: Covariates file (rows=covariates, cols=samples).
        interaction_path: Interaction file (cols=samples, col header=cos_t).

    Returns:
        (genotype_df, variant_df, phenotype_df, phenotype_pos_df,
         covariates_df, interaction_df)

        covariates_df: samples × covariates (transposed from file)
        interaction_df: samples × 1, indexed by sample IDs
    """
    # tensorqtl is imported here so the module can be imported without it
    # installed (e.g., during unit tests of validate_sample_alignment).
    try:
        from tensorqtl import genotypeio
        from tensorqtl.core import read_phenotype_bed
    except ImportError as e:
        raise ImportError(
            "tensorqtl is not installed. Install it with: pip install tensorqtl"
        ) from e

    print(f"Loading phenotypes from {phenotypes_path}")
    phenotype_df, phenotype_pos_df = read_phenotype_bed(phenotypes_path)

    print(f"Loading genotypes from {plink_prefix}")
    genotype_df, variant_df = genotypeio.load_genotypes(
        plink_prefix, select_samples=phenotype_df.columns
    )

    print(f"Loading covariates from {covariates_path}")
    # File format: rows=covariates, cols=samples → transpose to samples × covariates
    covariates_df = pd.read_csv(covariates_path, sep="\t", index_col=0).T

    print(f"Loading interaction terms from {interaction_path}")
    # File format: rows=samples, cols=interaction names (written by cosinor_preprocess.py)
    interaction_df = pd.read_csv(interaction_path, sep="\t", index_col=0)

    return genotype_df, variant_df, phenotype_df, phenotype_pos_df, covariates_df, interaction_df


def run_cosinor_mapping(
    genotype_df: pd.DataFrame,
    variant_df: pd.DataFrame,
    phenotype_df: pd.DataFrame,
    phenotype_pos_df: pd.DataFrame,
    covariates_df: pd.DataFrame,
    interaction_df: pd.DataFrame,
    prefix: str,
    output_dir: str,
    window: int = 1_000_000,
) -> None:
    """Run tensorQTL cis nominal mapping with the cos_t interaction term.

    Writes per-chromosome results to:
        <output_dir>/<prefix>.cis_qtl_pairs.<chr>.parquet
    and the top association per gene to:
        <output_dir>/<prefix>.cis_qtl_top_assoc.txt.gz

    Output columns follow standard tensorQTL format. Interaction-specific
    columns are labeled with the interaction term name ('cos_t'), e.g.:
        b_g_x_cos_t, b_g_x_cos_t_se, pval_g_x_cos_t
    """
    try:
        from tensorqtl import cis
    except ImportError as e:
        raise ImportError(
            "tensorqtl is not installed. Install it with: pip install tensorqtl"
        ) from e

    os.makedirs(output_dir, exist_ok=True)
    cis.map_nominal(
        genotype_df,
        variant_df,
        phenotype_df,
        phenotype_pos_df,
        prefix,
        covariates_df=covariates_df,
        interaction_df=interaction_df,
        window=window,
        output_dir=output_dir,
        write_top=True,
        write_stats=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run cosinor cis-eQTL interaction mapping using tensorQTL. "
            "Tests whether each SNP's effect on expression varies with the "
            "cosine of time-of-day (1-DF interaction test). "
            "Run cosinor_preprocess.py first to generate the required "
            "--covariates and --interaction files."
        )
    )
    parser.add_argument("--plink-prefix", required=True,
                        help="Prefix for PLINK2 pgen/psam/pvar files (no extension).")
    parser.add_argument("--phenotypes", required=True,
                        help="Expression BED file (.bed.gz or .bed.parquet).")
    parser.add_argument("--covariates", required=True,
                        help="Covariates file output by cosinor_preprocess.py.")
    parser.add_argument("--interaction", required=True,
                        help="Interaction file output by cosinor_preprocess.py (cos_t column).")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write result parquet files.")
    parser.add_argument("--prefix", required=True,
                        help="Output filename prefix (e.g. 'my_study.cosinor').")
    parser.add_argument("--window", type=int, default=1_000_000,
                        help="cis window size in bp (default: 1000000).")
    args = parser.parse_args()

    (genotype_df, variant_df, phenotype_df, phenotype_pos_df,
     covariates_df, interaction_df) = load_inputs(
        args.plink_prefix, args.phenotypes, args.covariates, args.interaction
    )

    validate_sample_alignment(phenotype_df, covariates_df, interaction_df)

    print(
        f"Running cosinor cis-QTL mapping (1-DF interaction: SNP × cos_t)\n"
        f"  phenotypes:        {phenotype_df.shape[0]}\n"
        f"  samples:           {phenotype_df.shape[1]}\n"
        f"  variants:          {variant_df.shape[0]}\n"
        f"  covariates:        {covariates_df.shape[1]}\n"
        f"  interaction terms: {list(interaction_df.columns)}"
    )

    run_cosinor_mapping(
        genotype_df, variant_df, phenotype_df, phenotype_pos_df,
        covariates_df, interaction_df,
        prefix=args.prefix,
        output_dir=args.output_dir,
        window=args.window,
    )
    print(
        f"Done. Results: {args.output_dir}/{args.prefix}.cis_qtl_pairs.*.parquet"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Confirm all unit tests still pass**

```bash
pytest tests/ -v
```

Expected: all tests PASSED (no regressions).

- [ ] **Step 3: Commit**

```bash
git add scripts/run_cosinor_qtl.py
git commit -m "feat: complete run_cosinor_qtl.py with load_inputs, run_cosinor_mapping, main()"
```

---

## Task 7: Smoke test against example data

This task verifies the full pipeline end-to-end using the GEUVADIS example data included in the repo. It requires tensorQTL and its dependencies to be installed (see `install/tensorqtl_env.yml`).

**Files:**
- No new files — runs scripts from the command line.

- [ ] **Step 1: Create a temporary metadata file for the 445 GEUVADIS samples**

```bash
python3 - <<'EOF'
import pandas as pd
# Read sample IDs from the covariates file
cov = pd.read_csv(
    "example/data/GEUVADIS.445_samples.covariates.txt",
    sep="\t", index_col=0
)
samples = list(cov.columns)
# Assign random hours (0–24) for smoke-test purposes
import numpy as np
np.random.seed(42)
hours = np.random.uniform(0, 24, size=len(samples))
meta = pd.DataFrame({"hour": hours}, index=samples)
meta.index.name = "sample_id"
meta.to_csv("/tmp/geuvadis_test_meta.tsv", sep="\t")
print(f"Wrote metadata for {len(samples)} samples")
EOF
```

Expected: `Wrote metadata for 445 samples`

- [ ] **Step 2: Run cosinor_preprocess.py on the example data**

```bash
python3 scripts/cosinor_preprocess.py \
  --metadata /tmp/geuvadis_test_meta.tsv \
  --covariates example/data/GEUVADIS.445_samples.covariates.txt \
  --out-covariates /tmp/geuvadis_cosinor_covariates.txt \
  --out-interaction /tmp/geuvadis_cos_t.txt \
  --time-col hour
```

Expected output (no error):
```
Loading metadata from /tmp/geuvadis_test_meta.tsv
Loading covariates from example/data/GEUVADIS.445_samples.covariates.txt
Wrote updated covariates (XX rows) to /tmp/geuvadis_cosinor_covariates.txt
Wrote interaction file (445 samples) to /tmp/geuvadis_cos_t.txt
```

- [ ] **Step 3: Verify cosinor outputs look correct**

```bash
python3 - <<'EOF'
import pandas as pd
cov = pd.read_csv("/tmp/geuvadis_cosinor_covariates.txt", sep="\t", index_col=0)
print("Last 3 covariate rows:", list(cov.index[-3:]))
assert "cos_t" in cov.index and "sin_t" in cov.index, "cos_t/sin_t missing!"

inter = pd.read_csv("/tmp/geuvadis_cos_t.txt", sep="\t", index_col=0)
print("Interaction columns:", list(inter.columns))
print("Interaction shape:", inter.shape)
assert inter.shape == (445, 1), f"Expected (445, 1), got {inter.shape}"
assert list(inter.columns) == ["cos_t"], "Wrong column name"
print("Outputs look correct.")
EOF
```

Expected: `Outputs look correct.`

- [ ] **Step 4: Run run_cosinor_qtl.py on the example data**

```bash
mkdir -p /tmp/cosinor_qtl_out

python3 scripts/run_cosinor_qtl.py \
  --plink-prefix example/data/GEUVADIS.445_samples.GRCh38.20170504.maf01.filtered.nodup.chr18 \
  --phenotypes example/data/GEUVADIS.445_samples.expression.bed.gz \
  --covariates /tmp/geuvadis_cosinor_covariates.txt \
  --interaction /tmp/geuvadis_cos_t.txt \
  --output-dir /tmp/cosinor_qtl_out \
  --prefix GEUVADIS.cosinor \
  --window 1000000
```

Expected: script completes without error, writes parquet files to `/tmp/cosinor_qtl_out/`.

- [ ] **Step 5: Verify output has interaction columns**

```bash
python3 - <<'EOF'
import os, pandas as pd
out_dir = "/tmp/cosinor_qtl_out"
parquets = [f for f in os.listdir(out_dir) if f.endswith(".parquet")]
print(f"Output parquet files: {parquets}")
df = pd.read_parquet(os.path.join(out_dir, parquets[0]))
print("Output columns:", list(df.columns))
# interaction columns contain the term name
assert any("cos_t" in c for c in df.columns), "No cos_t interaction columns found!"
print(f"Row count: {len(df)}")
print("Smoke test PASSED.")
EOF
```

Expected: `Smoke test PASSED.`

---

## Task 8: Documentation (`docs/cosinor_eqtl.md`)

**Files:**
- Create: `docs/cosinor_eqtl.md`

- [ ] **Step 1: Write `docs/cosinor_eqtl.md`**

```markdown
# Cosinor eQTL Mapping with tensorQTL

## Biological Rationale

Gene expression levels vary with time of day in many tissues, driven by circadian
transcriptional programs. When samples are collected across a range of collection
times, two distinct biological questions arise:

1. **Confounding control**: time-of-day affects expression globally; without
   controlling for it, circadian genes appear differentially expressed across any
   factor that correlates with collection time (batch, season, study phase).
   Solution: include cos_t and sin_t as covariates.

2. **Interaction testing**: does a SNP's effect on expression *depend on* the time
   of day? A genotype × circadian rhythm interaction would indicate that a variant
   affects the amplitude or phase of a circadian gene. This is the cosinor eQTL test.

The cosinor model encodes time of day as:

    cos_t = cos(2π × hour / 24)
    sin_t = sin(2π × hour / 24)

These two terms together represent any 24-hour sinusoidal pattern. Including both
as covariates removes linear circadian confounding. Using cos_t as an interaction
term tests whether the SNP effect is modulated by the cosine component.

## What Is Implemented: 1-DF Test

The model fitted per gene, per variant is:

    expression ~ SNP + cos_t + sin_t + SNP×cos_t + other_covariates

This tests a single null hypothesis: the coefficient of SNP×cos_t equals zero.
This is a 1-degree-of-freedom (1-DF) test. It is sensitive to interactions whose
phase aligns with cos_t (i.e., peak effect at midnight, trough at noon).

**Output columns** follow standard tensorQTL convention for interaction terms.
With one interaction named `cos_t`, tensorQTL adds columns named:
`b_g_x_cos_t`, `b_g_x_cos_t_se`, `pval_g_x_cos_t` (alongside the main-effect
columns `b_g`, `b_g_se`, `pval_g`).

## What Is Not Yet Implemented: 2-DF Test

The complete circadian interaction model includes *both* interaction terms:

    expression ~ SNP + cos_t + sin_t + SNP×cos_t + SNP×sin_t + other_covariates

Testing both simultaneously is a 2-DF test (joint null: both SNP×cos_t = 0 and
SNP×sin_t = 0). This is more powerful than two 1-DF tests and is agnostic to the
phase of the interaction.

The 2-DF test is not yet implemented. A combined test statistic requires combining
the t-statistics for SNP×cos_t and SNP×sin_t into a chi-squared statistic with 2 DF
as post-processing on the tensorQTL output. See "Upgrade Paths" below.

## Workflow

### Step 1: Prepare cosinor covariates

```bash
python scripts/cosinor_preprocess.py \
  --metadata sample_metadata.tsv \
  --covariates existing_covariates.txt \
  --out-covariates covariates_with_cosinor.txt \
  --out-interaction interaction_cos_t.txt \
  --time-col hour \
  --period 24.0
```

`sample_metadata.tsv` must be tab-separated with sample IDs in the first column
and a numeric time-of-day column (default name: `hour`, float 0–24).

### Step 2: Run cosinor eQTL mapping

```bash
python scripts/run_cosinor_qtl.py \
  --plink-prefix path/to/genotypes \
  --phenotypes expression.bed.gz \
  --covariates covariates_with_cosinor.txt \
  --interaction interaction_cos_t.txt \
  --output-dir results/ \
  --prefix my_study.cosinor \
  --window 1000000
```

## Upgrade Paths

### ISO 8601 timestamp support

Currently `cosinor_preprocess.py` expects numeric hours (e.g., `14.5`). To support
ISO 8601 timestamps (e.g., `2024-01-15T14:30:00`), change only the
`parse_time_to_hours` function in `scripts/cosinor_preprocess.py`:

```python
# Current (numeric hours):
def parse_time_to_hours(value: str) -> float:
    return float(value)

# Replace with (ISO 8601):
from datetime import datetime
def parse_time_to_hours(value: str) -> float:
    dt = datetime.fromisoformat(value)
    return dt.hour + dt.minute / 60.0 + dt.second / 3600.0
```

All downstream code consumes the returned float; no other changes are needed.

### 2-DF joint test upgrade

To test SNP×cos_t and SNP×sin_t jointly:

1. In `scripts/cosinor_preprocess.py`, change `make_interaction_df`:
   ```python
   # Current (1-DF):
   return cos_t.to_frame(name="cos_t")

   # Replace with (2-DF):
   return pd.DataFrame({"cos_t": cos_t, "sin_t": sin_t})
   ```

2. `run_cosinor_qtl.py` requires no changes — tensorQTL handles multiple
   interaction columns automatically and outputs t-statistics for each.

3. Add post-processing to combine the two t-statistics into a 2-DF chi-squared
   test. For each variant–gene pair with t-statistics `t1` (cos_t) and `t2`
   (sin_t) and degrees of freedom `dof`:
   ```python
   import scipy.stats as stats
   chi2 = t1**2 + t2**2     # approximate; exact requires the correlation between t1 and t2
   pval_2df = stats.chi2.sf(chi2, df=2)
   ```
   This approximation assumes t1 and t2 are independent, which holds approximately
   when cos_t and sin_t are uncorrelated across samples.
```

- [ ] **Step 2: Commit**

```bash
git add docs/cosinor_eqtl.md
git commit -m "docs: add cosinor_eqtl.md with rationale, 1-DF/2-DF distinction, upgrade paths"
```

---

## Task 9: Final cleanup commit

- [ ] **Step 1: Run the full test suite one last time**

```bash
pytest tests/ -v
```

Expected: all tests PASSED, no warnings about missing imports.

- [ ] **Step 2: Add `scripts/` and `tests/` to `.gitignore` exclusions if needed**

Check that `__pycache__` and `.pytest_cache` are ignored:
```bash
grep -E "pycache|pytest_cache" .gitignore || echo "__pycache__/" >> .gitignore && echo ".pytest_cache/" >> .gitignore
```

- [ ] **Step 3: Final commit**

```bash
git add .gitignore
git commit -m "chore: ensure pycache and pytest artifacts are gitignored"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] Preprocessing script with `parse_time_to_hours` stub → Task 2–4
- [x] `--metadata`, `--covariates`, `--out-covariates`, `--out-interaction`, `--time-col`, `--period` CLI args → Task 4
- [x] cos_t + sin_t appended to covariates; cos_t alone as interaction file → Tasks 3–4
- [x] tensorQTL runner via Python API (`cis.map_nominal` with `interaction_df`) → Task 6
- [x] `--plink-prefix`, `--phenotypes`, `--covariates`, `--interaction`, `--output-dir`, `--prefix`, `--window` CLI args → Task 6
- [x] ISO timestamp upgrade path documented and stub clearly marked → Tasks 2, 8
- [x] 2-DF upgrade path documented → Tasks 3, 8
- [x] Input validation with informative errors → Tasks 3, 4, 5
- [x] Two separate, independent scripts → separate files
- [x] Standard tensorQTL file format compatibility → verified in Tasks 4, 7
- [x] README/docstring with biological rationale, 1-DF vs 2-DF, upgrade paths → Task 8
- [x] Smoke test against example data → Task 7

**Placeholder scan:** No TBD, TODO, or "implement later" in code steps. All test assertions and implementation code is complete.

**Type consistency:** `compute_cosinor` returns a `tuple` of two `pd.Series`; `make_interaction_df` receives a `pd.Series` and returns `pd.DataFrame` — consistent across Tasks 2, 3, and all references in Task 4.
