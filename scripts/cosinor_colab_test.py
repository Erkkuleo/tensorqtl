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
# (torch is pre-installed in Colab; we add the rest)
"""
!pip install -q \
    pandas pyarrow scipy \
    "Pgenlib>=0.90.1" \
    pandas-plink \
    qtl
"""

# ============================================================
# Cell 2 — Clone the cosinor branch and add it to the path
# ============================================================
"""
import sys, os

REPO = "https://github.com/Erkkuleo/tensorqtl.git"
BRANCH = "feat/cosinor-eqtl"

!git clone --depth 1 --branch {BRANCH} {REPO} tensorqtl_repo

# Make the tensorqtl package and scripts importable
sys.path.insert(0, "/content/tensorqtl_repo")
sys.path.insert(0, "/content/tensorqtl_repo/tensorqtl")
os.chdir("/content/tensorqtl_repo")
print("Cloned. Working directory:", os.getcwd())
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
# Cell 5 — Run cosinor_preprocess.py
# ============================================================
"""
!python scripts/cosinor_preprocess.py \
    --metadata /tmp/meta.tsv \
    --covariates example/data/GEUVADIS.445_samples.covariates.txt \
    --out-covariates /tmp/cov_cosinor.txt \
    --out-interaction /tmp/interaction_cos_t.txt \
    --time-col hour

# Quick sanity check
cov_out = pd.read_csv("/tmp/cov_cosinor.txt", sep="\t", index_col=0)
int_out = pd.read_csv("/tmp/interaction_cos_t.txt", sep="\t", index_col=0)
print("Covariate rows:", list(cov_out.index[-3:]))   # should end with cos_t, sin_t
print("Interaction shape:", int_out.shape)            # should be (445, 1)
print("Interaction range: cos_t in [{:.3f}, {:.3f}]".format(
    int_out["cos_t"].min(), int_out["cos_t"].max()))
"""

# ============================================================
# Cell 6 — Run cosinor eQTL mapping  (~2–5 min on T4)
# ============================================================
"""
import os
os.makedirs("/tmp/cosinor_out", exist_ok=True)

!python scripts/run_cosinor_qtl.py \
    --plink-prefix example/data/GEUVADIS.445_samples.GRCh38.20170504.maf01.filtered.nodup.chr18 \
    --phenotypes example/data/GEUVADIS.445_samples.expression.bed.gz \
    --covariates /tmp/cov_cosinor.txt \
    --interaction /tmp/interaction_cos_t.txt \
    --output-dir /tmp/cosinor_out \
    --prefix GEUVADIS.cosinor \
    --window 1000000
"""

# ============================================================
# Cell 7 — Inspect results
# ============================================================
"""
import pandas as pd

df = pd.read_parquet("/tmp/cosinor_out/GEUVADIS.cosinor.cis_qtl_pairs.chr18.parquet")

print(f"Total variant-gene pairs tested: {len(df):,}")
print(f"Genes (phenotypes):              {df['phenotype_id'].nunique()}")
print(f"Variants:                        {df['variant_id'].nunique():,}")
print()
print("Columns:", df.columns.tolist())
print()

# Distribution of interaction p-values
import numpy as np
bins = [0, 0.001, 0.01, 0.05, 0.1, 1.0]
counts, _ = np.histogram(df["pval_g_x_cos_t"].dropna(), bins=bins)
print("pval_g_x_cos_t distribution (random hours → should be ~uniform):")
for lo, hi, n in zip(bins, bins[1:], counts):
    print(f"  [{lo:.3f}, {hi:.3f}): {n:,}")

print()
print("Top 10 interaction hits (SNP × cos_t):")
cols = ["phenotype_id", "variant_id", "af", "pval_g", "b_g_x_cos_t", "pval_g_x_cos_t"]
print(df.nsmallest(10, "pval_g_x_cos_t")[cols].to_string(index=False))
"""

# ============================================================
# Cell 8 — (Optional) Manhattan-style plot of interaction p-values
# ============================================================
"""
import matplotlib.pyplot as plt
import numpy as np

# -log10 p-values for interaction term
logp = -np.log10(df["pval_g_x_cos_t"].clip(lower=1e-300))
pos  = df["tss_distance"]   # distance from gene TSS

plt.figure(figsize=(14, 4))
plt.scatter(pos, logp, s=1, alpha=0.4, color="steelblue")
plt.axhline(-np.log10(0.05 / len(df)), color="red", lw=0.8,
            linestyle="--", label="Bonferroni 0.05")
plt.xlabel("TSS distance (bp)")
plt.ylabel("-log10(p) SNP × cos_t")
plt.title("Cosinor interaction eQTL — chr18 (random hours)")
plt.legend()
plt.tight_layout()
plt.show()
print("Note: with random hours the p-values should be uniform (no real signal).")
"""
