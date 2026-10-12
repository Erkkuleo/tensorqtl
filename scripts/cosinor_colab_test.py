"""
Cosinor eQTL — Google Colab test script
========================================
Paste each cell block into a separate Colab cell (Runtime > Run all, or Ctrl+Enter per cell).
A free T4 GPU is enough. Runtime type: Python 3 + GPU.

The script uses the GEUVADIS chr18 example data bundled in the repo
and assigns *random* collection times — replace with real hours for real results.
"""

# ============================================================
# Cell 1 — Install dependencies
# ============================================================
# NOTE: After this cell finishes, do Runtime → Restart session,
#       then run all cells again. The version pins fix a
#       numpy/pandas ABI mismatch that causes TypeError in Colab.
"""
!pip install -q -U \
    "numpy>=2.0.0" \
    "pandas>=2.2.2" \
    pyarrow scipy \
    "Pgenlib>=0.90.1" \
    pandas-plink \
    qtl \
    statsmodels pytest
print("Installation complete. Go to Runtime → Restart session, then re-run all cells.")
"""

# ============================================================
# Cell 2 — Clone the cosinor branch and add it to the path
# (safe to re-run: deletes old copy and reclones fresh)
# ============================================================
"""
import sys, os

REPO   = "https://github.com/Erkkuleo/tensorqtl.git"
BRANCH = "feat/cosinor-joint-wald"

!rm -rf /content/tensorqtl_repo
!git clone --depth 1 --branch {BRANCH} {REPO} /content/tensorqtl_repo

sys.path.insert(0, "/content/tensorqtl_repo")
sys.path.insert(0, "/content/tensorqtl_repo/tensorqtl")
os.chdir("/content/tensorqtl_repo")
print("Working directory:", os.getcwd())
"""

# ============================================================
# Cell 3 — Check GPU
# ============================================================
"""
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
"""

# ============================================================
# Cell 4 — Create fake time-of-day metadata for 445 samples
#           (swap in your real hours here)
# ============================================================
"""
import pandas as pd
import numpy as np

# Read sample IDs from the header line directly — avoids a pandas 2.x
# crash that affects files with many "NA..."-prefixed column names.
with open("example/data/GEUVADIS.445_samples.covariates.txt") as f:
    sample_ids = f.readline().rstrip("\n").split("\t")[1:]

print(f"Found {len(sample_ids)} samples")

np.random.seed(42)
hours = np.random.uniform(0, 24, size=len(sample_ids))

meta = pd.DataFrame({"hour": hours}, index=sample_ids)
meta.index.name = "sample_id"
meta.to_csv("/tmp/meta.tsv", sep="\t")
print(f"Created metadata for {len(meta)} samples")
print(meta.describe())
"""

# ============================================================
# Cell 5 — Unit tests (CPU-only tests, ~1–3 min)
#   Includes the statsmodels exactness check, null calibration,
#   phase invariance and power tests for the joint cosinor tests.
# ============================================================
"""
!python -m pytest tests/ -v
"""

# ============================================================
# Cell 6 — Backward compatibility: compare map_nominal output to master
#           (needs a separate checkout of master)
# ============================================================
"""
!rm -rf /content/tensorqtl_master
!git clone --depth 1 --branch master {REPO} /content/tensorqtl_master
!TENSORQTL_MASTER_DIR=/content/tensorqtl_master python -m pytest tests/test_backward_compat.py -v
"""

# ============================================================
# Cell 7 — Run cosinor_preprocess.py (both harmonics, for 4df/5df)
# ============================================================
"""
!python scripts/cosinor_preprocess.py \
    --metadata /tmp/meta.tsv \
    --covariates example/data/GEUVADIS.445_samples.covariates.txt \
    --out-covariates /tmp/cov_cosinor.txt \
    --out-interaction /tmp/interaction_cosinor.txt \
    --time-col hour \
    --harmonics 2

# Quick sanity check
cov_out = pd.read_csv("/tmp/cov_cosinor.txt", sep="\t", index_col=0)
int_out = pd.read_csv("/tmp/interaction_cosinor.txt", sep="\t", index_col=0)
print("Covariate rows (no cosinor terms):", list(cov_out.index[-3:]))
print("Interaction columns:", list(int_out.columns))  # cos_t, sin_t, cos2_t, sin2_t
print("Interaction shape:", int_out.shape)            # (445, 4)
"""

# ============================================================
# Cell 8 — Run cosinor eQTL mapping with joint tests (~2–5 min on T4)
# ============================================================
"""
import os
os.makedirs("/tmp/cosinor_out", exist_ok=True)

!time python scripts/run_cosinor_qtl.py \
    --plink-prefix example/data/GEUVADIS.445_samples.GRCh38.20170504.maf01.filtered.nodup.chr18 \
    --phenotypes example/data/GEUVADIS.445_samples.expression.bed.gz \
    --covariates /tmp/cov_cosinor.txt \
    --interaction /tmp/interaction_cosinor.txt \
    --output-dir /tmp/cosinor_out \
    --prefix GEUVADIS.cosinor \
    --window 1000000 \
    --tests 2df,4df,5df \
    --also-acat
"""

# ============================================================
# Cell 9 — Inspect results
# ============================================================
"""
import numpy as np
import pandas as pd

df = pd.read_parquet("/tmp/cosinor_out/GEUVADIS.cosinor.cis_qtl_pairs.chr18.parquet")
top = pd.read_csv("/tmp/cosinor_out/GEUVADIS.cosinor.cis_qtl_top_assoc.txt.gz", sep="\t", index_col=0)

print(f"Total variant-gene pairs tested: {len(df):,}")
print(f"Genes (phenotypes):              {df['phenotype_id'].nunique()}")
print()
print("Pair columns:", df.columns.tolist())
print("Top columns: ", top.columns.tolist())
print()

# With random hours the interaction tests should be ~uniform (5df is not: it includes the eQTL main effect)
bins = [0, 0.001, 0.01, 0.05, 0.1, 1.0]
for col in ["pval_cosinor_2df_h2", "pval_cosinor_4df_h2", "pval_cosinor_5df_h2", "pval_g_x_cosinor_acat"]:
    counts, _ = np.histogram(df[col].dropna(), bins=bins)
    print(col, " fraction < 0.05:", f"{(df[col] < 0.05).mean():.4f}")
    for lo, hi, n in zip(bins, bins[1:], counts):
        print(f"  [{lo:.3f}, {hi:.3f}): {n:,}")

print()
print("Top 10 genes by 2-DF joint test:")
cols = ["variant_id", "num_var", "pval_g", "pval_cosinor_2df_h2",
        "pval_cosinor_2df_h2_bonf", "qval_cosinor_2df_h2", "pval_cosinor_4df_h2"]
print(top.nsmallest(10, "pval_cosinor_2df_h2")[cols].to_string())
"""

# ============================================================
# Cell 10 — QQ plots of the joint tests
# ============================================================
"""
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for ax, col in zip(axes, ["pval_g", "pval_cosinor_2df_h2", "pval_cosinor_4df_h2", "pval_g_x_cosinor_acat"]):
    pv = np.sort(df[col].dropna().values)
    exp = -np.log10(np.arange(1, len(pv) + 1) / (len(pv) + 1))
    ax.scatter(exp, -np.log10(pv.clip(1e-300)), s=1, alpha=0.4)
    ax.plot([0, exp.max()], [0, exp.max()], "r--", lw=0.8)
    ax.set_title(col)
    ax.set_xlabel("Expected -log10(p)")
    ax.set_ylabel("Observed -log10(p)")
plt.tight_layout()
plt.show()
print("With random hours, 2df/4df should follow the diagonal; pval_g and 5df will not (real eQTLs).")
"""

# ============================================================
# Cell 11 — Runtime: joint tests vs the master code path
# ============================================================
"""
!python scripts/benchmark_joint_wald.py \
    --plink-prefix example/data/GEUVADIS.445_samples.GRCh38.20170504.maf01.filtered.nodup.chr18 \
    --phenotypes example/data/GEUVADIS.445_samples.expression.bed.gz \
    --covariates example/data/GEUVADIS.445_samples.covariates.txt \
    --chrom chr18
"""
