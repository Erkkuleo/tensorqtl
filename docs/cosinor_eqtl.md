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

**Output columns** for a single interaction term named `cos_t`:

| Column | Description |
|---|---|
| `pval_g`, `b_g`, `b_g_se` | Main SNP effect |
| `pval_cos_t`, `b_cos_t`, `b_cos_t_se` | Main cos_t effect |
| `pval_g_x_cos_t`, `b_g_x_cos_t`, `b_g_x_cos_t_se` | SNP × cos_t interaction |

## What Is Not Yet Implemented: 2-DF Test

The complete circadian interaction model includes *both* interaction terms:

    expression ~ SNP + cos_t + sin_t + SNP×cos_t + SNP×sin_t + other_covariates

Testing both simultaneously is a 2-DF test (joint null: both SNP×cos_t = 0 and
SNP×sin_t = 0). This is more powerful than two 1-DF tests and is agnostic to the
phase of the interaction. See "Upgrade Paths" below.

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
   chi2 = t1**2 + t2**2     # approximate; assumes t1 and t2 are independent
   pval_2df = stats.chi2.sf(chi2, df=2)
   ```
   This approximation holds when cos_t and sin_t are uncorrelated across samples,
   which is approximately true for typical study designs.
