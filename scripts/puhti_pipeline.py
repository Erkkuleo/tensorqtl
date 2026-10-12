#!/usr/bin/env python3
"""
Cosinor eQTL full pipeline for Puhti supercomputer (CSC Finland).

Runs all steps from raw GEUVADIS data to final plots across all 22 chromosomes.
Uses SLURM for GPU-accelerated tensorQTL mapping and CHIRAL for phase inference.

SETUP ON PUHTI (run once before this script):
  ssh <username>@puhti.csc.fi
  cd /scratch/<project>/
  git clone https://github.com/Erkkuleo/tensorqtl.git
  cd tensorqtl

  # Create conda environment
  module load tykky
  conda-containerize new --mamba --prefix /projappl/<project>/cosinor-env install/tensorqtl_env.yml

  # Install extra Python packages
  /projappl/<project>/cosinor-env/bin/pip install scikit-learn

  # Download plink2
  wget https://s3.amazonaws.com/plink2-assets/alpha6/plink2_linux_avx2_20260425.zip
  unzip plink2_linux_avx2_20260425.zip
  mv plink2 /projappl/<project>/cosinor-env/bin/

USAGE:
  # Full pipeline (all steps)
  python scripts/puhti_pipeline.py --project <project_id> --all

  # Individual steps
  python scripts/puhti_pipeline.py --project <project_id> --step download
  python scripts/puhti_pipeline.py --project <project_id> --step chiral
  python scripts/puhti_pipeline.py --project <project_id> --step preprocess
  python scripts/puhti_pipeline.py --project <project_id> --step map
  python scripts/puhti_pipeline.py --project <project_id> --step plots

  # Only clock gene chromosomes (fast test)
  python scripts/puhti_pipeline.py --project <project_id> --step map --chroms 12,17

  # Joint tests (default: --tests 2df). 4df/5df need the 12 h harmonic:
  python scripts/puhti_pipeline.py --project <project_id> --all --harmonics 2 --tests 2df,4df
"""
import argparse
import gzip
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import lstsq
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── URLs ────────────────────────────────────────────────────────────────────
EBI_BASE = "http://ftp.ebi.ac.uk/pub/databases/microarray/data/experiment/GEUV/E-GEUV-1"
RPKM_URL = f"{EBI_BASE}/analysis_results/GD462.GeneQuantRPKM.50FN.samplename.resk10.txt.gz"
SDRF_URL = "http://www.ebi.ac.uk/arrayexpress/files/E-GEUV-1/E-GEUV-1.sdrf.txt"
GTF_URL  = ("https://ftp.ensembl.org/pub/release-111/gtf/homo_sapiens/"
            "Homo_sapiens.GRCh38.111.chr.gtf.gz")
VCF_BASE = ("https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/"
            "1000G_2504_high_coverage/working/20201028_3202_phased")

ALL_CHROMS = [str(i) for i in range(1, 23)]
VALID_TESTS = ["2df", "3df", "4df", "5df"]
HARMONIC2_TESTS = {"4df", "5df"}  # tests that need cos2_t/sin2_t (see tensorqtl/cosinor.py)


def joint_labels(tests):
    """Output labels of the joint tests, e.g. ['cosinor_2df_h2', 'cosinor_4df_h2']."""
    k = 2 if HARMONIC2_TESTS.intersection(tests) else 1
    return [f"cosinor_{t}_h{k}" for t in tests]


def padjust_bh(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    o = np.argsort(p)[::-1]
    q = np.minimum(1, np.minimum.accumulate(n / np.arange(n, 0, -1) * p[o]))
    out = np.empty(n)
    out[o] = q
    return out


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def run(cmd, **kwargs):
    print(f"  $ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    subprocess.run(cmd, check=True, shell=isinstance(cmd, str), **kwargs)


def download(url, dest):
    if Path(dest).exists():
        print(f"  Already exists: {dest}")
        return
    print(f"  Downloading {url}")
    run(["wget", "-q", "-O", str(dest), url])


def inverse_normal_transform(x):
    ranks = stats.rankdata(x, method="average")
    return stats.norm.ppf(ranks / (len(ranks) + 1))


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — DOWNLOAD
# ════════════════════════════════════════════════════════════════════════════

def step_download(dirs):
    print("\n=== STEP 1: Download GEUVADIS expression + annotation ===")
    download(RPKM_URL, dirs["raw"] / "GD462.GeneQuantRPKM.txt.gz")
    download(SDRF_URL, dirs["raw"] / "E-GEUV-1.sdrf.txt")
    download(GTF_URL,  dirs["raw"] / "Homo_sapiens.GRCh38.111.chr.gtf.gz")

    # Parse EUR samples
    sdrf = pd.read_csv(dirs["raw"] / "E-GEUV-1.sdrf.txt", sep="\t")
    sdrf = sdrf.drop_duplicates(subset=["Source Name"])
    eur_pops = ["British", "Finnish", "Utah", "Tuscan"]
    eur = sdrf[sdrf["Characteristics[ancestry category]"].isin(eur_pops)]["Source Name"]
    eur.to_csv(dirs["raw"] / "samples_EUR.txt", index=False, header=False)
    print(f"  EUR samples: {len(eur)}")

    # Load, filter, normalize expression
    rpkm = pd.read_csv(dirs["raw"] / "GD462.GeneQuantRPKM.txt.gz", sep="\t", index_col=0)
    rpkm = rpkm[[c for c in rpkm.columns if c in set(eur)]]
    rpkm.clip(lower=0, inplace=True)
    rpkm.to_csv(dirs["raw"] / "rpkm_EUR.tsv.gz", sep="\t", compression="gzip")
    print(f"  Raw RPKM saved ({rpkm.shape})")

    # Normalize for eQTL mapping
    expressed = (rpkm > 0.1).sum(axis=1) >= int(0.1 * rpkm.shape[1])
    log_rpkm  = np.log2(rpkm.loc[expressed].clip(lower=0) + 1)
    log_rpkm  = log_rpkm.replace([np.inf, -np.inf], np.nan).dropna()
    normed    = log_rpkm.apply(lambda r: inverse_normal_transform(r.values),
                               axis=1, result_type="expand")
    normed.columns = log_rpkm.columns

    # Build BED with genomic coordinates
    print("  Parsing GTF for gene coordinates...")
    coords = {}
    with gzip.open(dirs["raw"] / "Homo_sapiens.GRCh38.111.chr.gtf.gz", "rt") as f:
        for line in f:
            if line.startswith("#") or "\tgene\t" not in line:
                continue
            fields = line.split("\t")
            chrom  = fields[0]
            if chrom.startswith("CHR") or "_" in chrom:
                continue
            chrom = f"chr{chrom}" if not chrom.startswith("chr") else chrom
            gid   = None
            for attr in fields[8].split(";"):
                attr = attr.strip()
                if attr.startswith("gene_id"):
                    gid = attr.split('"')[1].split(".")[0]
                    break
            if gid:
                coords[gid] = (chrom, int(fields[3]) - 1, int(fields[4]))

    rows = []
    for gene in normed.index:
        base = gene.split(".")[0]
        if base in coords:
            rows.append(coords[base] + (gene,))

    coord_df = pd.DataFrame(rows, columns=["#chr", "start", "end", "gene_id"])
    chrom_rank = {f"chr{i}": i for i in range(1, 23)}
    chrom_rank.update({"chrX": 23, "chrY": 24, "chrMT": 25})
    coord_df["_r"] = coord_df["#chr"].map(lambda c: chrom_rank.get(c, 99))
    coord_df = coord_df.sort_values(["_r", "start", "end"]).drop(columns=["_r"])

    # Explicit join to avoid index name loss
    coord_idx = coord_df.set_index("gene_id")
    shared    = coord_idx.index.intersection(normed.index)
    bed       = pd.concat([coord_idx.loc[shared],
                           normed.reindex(shared)], axis=1)
    bed.index.name = "gene_id"
    bed = bed.reset_index()
    sample_cols = [c for c in normed.columns if c in bed.columns]
    bed = bed[["#chr", "start", "end", "gene_id"] + sample_cols]
    bed.to_csv(dirs["data"] / "expression.bed.gz", sep="\t", index=False,
               compression="gzip")
    print(f"  BED written ({len(bed)} genes)")

    # Expression PCs as covariates
    X    = StandardScaler().fit_transform(normed.values.T)
    pcs  = PCA(n_components=20).fit_transform(X)
    pc_df = pd.DataFrame(pcs.T,
                         index=[f"ExprPC{i+1}" for i in range(20)],
                         columns=normed.columns)
    pc_df.to_csv(dirs["data"] / "covariates_base.txt", sep="\t")
    print("  Covariates written (20 expression PCs)")


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — CHIRAL PHASE INFERENCE
# ════════════════════════════════════════════════════════════════════════════

CHIRAL_R = r"""
# Add personal R library (pre-installed on login node) to search path
personal_lib <- file.path(Sys.getenv("HOME"), "R", "library")
if (dir.exists(personal_lib)) .libPaths(c(personal_lib, .libPaths()))

if (!requireNamespace("CHIRAL", quietly=TRUE)) {
  if (!requireNamespace("remotes", quietly=TRUE))
    install.packages("remotes", repos="https://cloud.r-project.org",
                     lib=personal_lib, quiet=TRUE)
  remotes::install_github("naef-lab/CHIRAL/Pkg/CHIRAL",
                           lib=personal_lib, quiet=TRUE)
}
library(CHIRAL)
args       <- commandArgs(trailingOnly=TRUE)
rpkm_path  <- args[1]; out_path <- args[2]; n_iter <- as.integer(args[3])

E <- as.matrix(read.table(rpkm_path, sep="\t", header=TRUE,
                           row.names=1, check.names=FALSE))
E[E < 0] <- 0
E <- log2(E + 1)
E <- E[apply(E, 1, function(x) all(is.finite(x))), ]

# Auto-detect gene ID format and supply matching clock genes
hgnc <- c("ARNTL","CLOCK","CRY1","CRY2","PER1","PER2","PER3","NR1D1","NR1D2","RORA")
ensg <- c("ENSG00000133794","ENSG00000049246","ENSG00000008405",
          "ENSG00000113163","ENSG00000179094","ENSG00000132326",
          "ENSG00000049246","ENSG00000069667","ENSG00000174738","ENSG00000069667")
if (sum(hgnc %in% rownames(E)) >= 3) {
  clk <- hgnc[hgnc %in% rownames(E)]
} else {
  clk <- rownames(E)[sapply(rownames(E),
           function(g) any(startsWith(g, ensg)))]
  if (length(clk) < 3) clk <- rownames(E)
}
cat(sprintf("Clock genes for init: %d\n", length(clk)))
result <- CHIRAL(E, iterations=n_iter, clockgenes=clk,
                 mean.centre.E=TRUE, TSM=TRUE, pbar=TRUE)
phi    <- result$phi
hours  <- phi * 24 / (2*pi)
sids   <- if (!is.null(names(phi)) && length(names(phi))==length(phi)) names(phi) else colnames(E)
out    <- data.frame(sample_id=sids, hour=round(hours,4),
                     phase_radians=round(phi,6), row.names=NULL)
write.table(out, out_path, sep="\t", row.names=FALSE, quote=FALSE)
cat(sprintf("Phases written to %s\n", out_path))
"""


def step_chiral(dirs, n_iter=500, n_top=3000):
    print("\n=== STEP 2: CHIRAL phase inference ===")
    rpkm = pd.read_csv(dirs["raw"] / "rpkm_EUR.tsv.gz", sep="\t",
                       index_col=0, compression="gzip")

    # Select top variable genes
    rpkm.clip(lower=0, inplace=True)
    log  = np.log2(rpkm + 1)
    expr = (rpkm > 1).sum(axis=1) >= int(0.2 * rpkm.shape[1])
    log  = log.loc[expr]
    cv   = log.std(axis=1) / (log.mean(axis=1) + 1e-6)
    top  = cv.nlargest(min(n_top, len(cv))).index

    tmp_expr = dirs["tmp"] / "rpkm_chiral.tsv"
    rpkm.loc[top].to_csv(tmp_expr, sep="\t")

    tmp_r = dirs["tmp"] / "chiral.R"
    tmp_r.write_text(CHIRAL_R)

    out_phases = dirs["data"] / "chiral_phases.tsv"

    # Wrap in a bash script that loads r-env so apptainer_wrapper is available
    # (r-env on Puhti uses Apptainer containers and needs the full module env)
    # Submit CHIRAL as a SLURM job — pytorch and r-env use separate Apptainer
    # containers and cannot call each other from within subprocess.
    # A dedicated SLURM job loads only r-env, avoiding the container conflict.
    slurm_script = dirs["tmp"] / "chiral_job.sh"
    slurm_script.write_text(textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name=chiral_phase
        #SBATCH --account={os.environ.get('SLURM_JOB_ACCOUNT', 'project_2013539')}
        #SBATCH --partition=small
        #SBATCH --time=01:00:00
        #SBATCH --mem=64G
        #SBATCH --nodes=1
        #SBATCH --ntasks=1
        #SBATCH --cpus-per-task=4
        #SBATCH --output={dirs['logs']}/chiral_%j.out
        #SBATCH --error={dirs['logs']}/chiral_%j.err

        module load r-env

        # srun is required on Puhti for r-env Apptainer containers in batch jobs
        srun Rscript {tmp_r} {tmp_expr} {out_phases} {n_iter}
    """))

    if out_phases.exists():
        print(f"  Phases already exist: {out_phases}")
        return

    print(f"\n  CHIRAL data prepared. pytorch and r-env are in separate containers")
    print(f"  and cannot call each other. Submit the job from the LOGIN NODE:\n")
    print(f"    sbatch {slurm_script}\n")
    print(f"  Monitor with:  squeue -u $USER")
    print(f"  When done, re-run this step to continue:")
    print(f"    python3 scripts/puhti_pipeline.py --project project_2013539 --step chiral")
    print(f"\n  Output will be written to: {out_phases}")
    sys.exit(0)

    phases = pd.read_csv(out_phases, sep="\t")
    print(f"  Samples: {len(phases)}")
    print(f"  Hour SD: {phases['hour'].std():.2f}  (6.93 = uniform)")


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — COSINOR PREPROCESS
# ════════════════════════════════════════════════════════════════════════════

def step_preprocess(dirs, harmonics=1):
    print(f"\n=== STEP 3: Cosinor preprocessing (harmonics = {harmonics}) ===")
    run([
        sys.executable, "scripts/cosinor_preprocess.py",
        "--metadata",        str(dirs["data"] / "chiral_phases.tsv"),
        "--covariates",      str(dirs["data"] / "covariates_base.txt"),
        "--out-covariates",  str(dirs["data"] / "covariates_cosinor.txt"),
        "--out-interaction", str(dirs["data"] / "interaction.txt"),
        "--time-col", "hour", "--mode", "tod",
        "--harmonics", str(harmonics),
    ])


# ════════════════════════════════════════════════════════════════════════════
# STEP 4 — PER-CHROMOSOME MAPPING (SLURM or local)
# ════════════════════════════════════════════════════════════════════════════

SLURM_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name=cosinor_chr{chrom}
#SBATCH --account={project}
#SBATCH --partition=small
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output={log_dir}/chr{chrom}_%j.out
#SBATCH --error={log_dir}/chr{chrom}_%j.err

module load pytorch   # includes python3.12, torch+CUDA, pandas, scipy, sklearn
module load r-env     # R for CHIRAL (only needed if running CHIRAL in the job)
export PATH=/users/$USER/.local/bin:/scratch/{project}/Erkka/tensorqtl:$PATH

cd {workdir}

VCF_URL="{vcf_base}/CCDG_14151_B01_GRM_WGS_2020-08-05_chr{chrom}.filtered.shapeit2-duohmm-phased.vcf.gz"
VCF="{geno_dir}/chr{chrom}.vcf.gz"
PGEN="{geno_dir}/chr{chrom}"

# Download VCF if needed
if [ ! -f "${{PGEN}}.pgen" ]; then
  wget -q -O "$VCF" "$VCF_URL"
  plink2 --vcf "$VCF" \\
         --keep {samples_file} \\
         --maf 0.01 \\
         --make-pgen \\
         --output-chr chrM \\
         --out "$PGEN" \\
         --threads 4
  rm -f "$VCF"
fi

# Run cosinor eQTL mapping
python3 scripts/run_cosinor_qtl.py \\
  --plink-prefix "$PGEN" \\
  --phenotypes   {data_dir}/expression.bed.gz \\
  --covariates   {data_dir}/covariates_cosinor.txt \\
  --interaction  {data_dir}/interaction.txt \\
  --output-dir   {out_dir} \\
  --prefix       geuvadis.chiral.chr{chrom} \\
  --tests        {tests}
"""


def step_map(dirs, project, chroms=ALL_CHROMS, use_slurm=True, tests=("2df",)):
    print(f"\n=== STEP 4: cis-eQTL mapping ({len(chroms)} chromosomes, tests: {','.join(tests)}) ===")
    workdir = Path.cwd()

    for chrom in chroms:
        # one prefix per chromosome so the top-association tables are not overwritten
        out_file = dirs["out"] / f"geuvadis.chiral.chr{chrom}.cis_qtl_pairs.chr{chrom}.parquet"
        if out_file.exists():
            print(f"  chr{chrom}: already done, skipping")
            continue

        if use_slurm:
            script = SLURM_TEMPLATE.format(
                chrom=chrom, project=project,
                log_dir=dirs["logs"], workdir=workdir,
                vcf_base=VCF_BASE,
                geno_dir=dirs["geno"], data_dir=dirs["data"],
                out_dir=dirs["out"],
                samples_file=dirs["raw"] / "samples_EUR.txt",
                tests=",".join(tests),
            )
            slurm_path = dirs["tmp"] / f"slurm_chr{chrom}.sh"
            slurm_path.write_text(script)
            result = subprocess.run(["sbatch", str(slurm_path)],
                                    capture_output=True, text=True)
            print(f"  chr{chrom}: {result.stdout.strip()}")
        else:
            _map_local(dirs, chrom, tests=tests)

    if use_slurm:
        print("\n  Jobs submitted. Monitor with: squeue -u $USER")
        print("  When all done, run: python scripts/puhti_pipeline.py --step plots")


def _map_local(dirs, chrom, tests=("2df",)):
    """Download VCF, convert to pgen, run mapping (for login node / interactive)."""
    vcf_url = (f"{VCF_BASE}/CCDG_14151_B01_GRM_WGS_2020-08-05_"
               f"chr{chrom}.filtered.shapeit2-duohmm-phased.vcf.gz")
    vcf     = dirs["geno"] / f"chr{chrom}.vcf.gz"
    pgen    = dirs["geno"] / f"chr{chrom}"

    if not (pgen.with_suffix(".pgen")).exists():
        print(f"  Downloading chr{chrom} VCF...")
        download(vcf_url, vcf)
        run(["plink2", "--vcf", str(vcf),
             "--keep", str(dirs["raw"] / "samples_EUR.txt"),
             "--maf", "0.01", "--make-pgen",
             "--output-chr", "chrM",
             "--out", str(pgen), "--threads", "4"])
        vcf.unlink(missing_ok=True)

    result = subprocess.run([
        sys.executable, "scripts/run_cosinor_qtl.py",
        "--plink-prefix",  str(pgen),
        "--phenotypes",    str(dirs["data"] / "expression.bed.gz"),
        "--covariates",    str(dirs["data"] / "covariates_cosinor.txt"),
        "--interaction",   str(dirs["data"] / "interaction.txt"),
        "--output-dir",    str(dirs["out"]),
        "--prefix",        f"geuvadis.chiral.chr{chrom}",
        "--tests",         ",".join(tests),
    ], capture_output=True, text=True)
    print(result.stdout[-2000:])
    if result.returncode != 0:
        print("STDERR:", result.stderr[-1000:])
        raise RuntimeError(f"Mapping failed for chr{chrom}")

    for ext in [".pgen", ".psam", ".pvar"]:
        p = pgen.with_suffix(ext)
        if p.exists():
            p.unlink()


# ════════════════════════════════════════════════════════════════════════════
# STEP 5 — PLOTS & SUMMARY
# ════════════════════════════════════════════════════════════════════════════

def _qq_panel(ax, pv, title):
    pv  = pv[np.isfinite(pv)]
    n   = len(pv)
    obs = -np.log10(np.sort(pv).clip(1e-300))
    exp = -np.log10(np.arange(1, n+1) / (n+1))
    # 95% CI
    idx = np.arange(1, n+1)
    lo  = -np.log10(stats.beta.ppf(0.975, idx, n-idx+1).clip(1e-300))
    hi  = -np.log10(stats.beta.ppf(0.025, idx, n-idx+1).clip(1e-300))
    ax.fill_between(exp, lo, hi, color="grey", alpha=0.2)
    ax.scatter(exp, obs, s=1, alpha=0.3, color="steelblue")
    ax.plot([0, exp.max()], [0, exp.max()], "r--", lw=0.8)
    # Lambda GC (median of 1-DF chi-squared quantiles; valid for any p-value)
    chi2 = stats.chi2.ppf(1 - pv.clip(1e-300, 1), df=1)
    lam  = np.median(chi2) / stats.chi2.ppf(0.5, df=1)
    ax.set_title(f"{title}\nλ={lam:.3f}", fontsize=10)
    ax.set_xlabel("Expected -log10(p)")
    ax.set_ylabel("Observed -log10(p)")


def step_plots(dirs, tests=("2df",)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("\n=== STEP 5: Generating plots and summary ===")
    labels = joint_labels(tests)

    # Load all parquets
    parquets = sorted(dirs["out"].glob("geuvadis.chiral.chr*.cis_qtl_pairs.chr*.parquet"))
    if not parquets:
        print("  No parquet files found. Run mapping first.")
        return

    frames = [pd.read_parquet(p) for p in parquets]
    df     = pd.concat(frames, ignore_index=True)
    print(f"  Total variant-gene pairs: {len(df):,}")
    print(f"  Chromosomes:              {len(parquets)}")
    missing = [f"pval_{l}" for l in labels if f"pval_{l}" not in df]
    if missing:
        print(f"  ERROR: columns {missing} not found. Re-run mapping with --tests {','.join(tests)} "
              f"(and --harmonics 2 for 4df/5df).")
        return

    # ── QQ plots: main eQTL + one panel per joint test ─────────────────────
    panels = [("pval_g", "Main eQTL (pval_g)")] + \
             [(f"pval_{l}", f"{t.upper()} joint cosinor ({l})") for t, l in zip(tests, labels)]
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5), squeeze=False)
    for ax, (col, title) in zip(axes[0], panels):
        _qq_panel(ax, df[col].dropna().values, title)
    plt.tight_layout()
    plt.savefig(dirs["plots"] / "qq_plots.png", dpi=150)
    plt.close()
    print("  Saved qq_plots.png")

    # ── Manhattan plots: one per joint test ────────────────────────────────
    chrom_rank = {f"chr{i}": i for i in range(1, 23)}
    chrom_rank.update({"chrX": 23})
    df["_cr"]  = df.get("variant_id", pd.Series(dtype=str)).apply(
        lambda v: chrom_rank.get("chr" + v.split("_")[0].replace("chr",""), 99)
        if isinstance(v, str) else 99
    )
    chroms_present = sorted(df["_cr"].unique())
    bonf = 0.05 / len(df)

    for t, label in zip(tests, labels):
        # Use start_distance as x-axis proxy (relative to gene)
        fig, ax = plt.subplots(figsize=(18, 5))
        colors  = ["steelblue", "cornflowerblue"]
        offset  = 0
        xticks, xlabels = [], []
        for i, cr in enumerate(chroms_present):
            sub  = df[df["_cr"] == cr]
            xpos = sub["start_distance"].fillna(0) + offset
            logp = -np.log10(sub[f"pval_{label}"].clip(1e-300))
            ax.scatter(xpos, logp, s=1, alpha=0.3, color=colors[i % 2])
            xticks.append(xpos.mean())
            xlabels.append(str(cr) if cr <= 22 else "X")
            offset = xpos.max() + 5_000_000
        ax.axhline(-np.log10(bonf), color="red", lw=0.8, linestyle="--",
                   label=f"Bonferroni p={bonf:.1e}")
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontsize=7)
        ax.set_xlabel("Chromosome")
        ax.set_ylabel(f"-log10(p)  SNP × cosinor ({t.upper()})")
        ax.set_title(f"Cosinor interaction eQTL ({label}) — GEUVADIS, CHIRAL phases")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(dirs["plots"] / f"manhattan_{t}.png", dpi=150)
        plt.close()
        print(f"  Saved manhattan_{t}.png")

    # ── Phase distribution ─────────────────────────────────────────────────
    phases = pd.read_csv(dirs["data"] / "chiral_phases.tsv", sep="\t")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(phases["hour"], bins=24, range=(0, 24), color="steelblue",
            edgecolor="white", alpha=0.8)
    ax.set_xlabel("Inferred hour (CHIRAL)")
    ax.set_ylabel("Samples")
    ax.set_title(f"Circadian phase distribution (n={len(phases)}, "
                 f"SD={phases['hour'].std():.2f} h)")
    plt.tight_layout()
    plt.savefig(dirs["plots"] / "phase_distribution.png", dpi=150)
    plt.close()
    print("  Saved phase_distribution.png")

    # ── Top hits table (ranked by the first test) ──────────────────────────
    rank_col = f"pval_{labels[0]}"
    cols = ["phenotype_id", "variant_id", "af", "pval_g",
            "b_g_x_cos_t", "pval_g_x_cos_t", "b_g_x_sin_t", "pval_g_x_sin_t"]
    cols += [f"pval_{l}" for l in labels]
    top = df.nsmallest(50, rank_col)[[c for c in cols if c in df]]
    top.to_csv(dirs["out"] / "top50_cosinor_hits.tsv", sep="\t", index=False)
    print("  Saved top50_cosinor_hits.tsv")
    print(f"\nTop 10 hits (by {rank_col}):")
    print(top.head(10)[["phenotype_id", "variant_id", "pval_g"] + [f"pval_{l}" for l in labels]]
          .to_string(index=False))

    # ── Gene-level results: genome-wide BH on Bonferroni-adjusted top p ────
    top_files = sorted(dirs["out"].glob("geuvadis.chiral.chr*.cis_qtl_top_assoc.txt.gz"))
    genes = pd.concat([pd.read_csv(f, sep="\t", index_col=0) for f in top_files]) if top_files else None
    if genes is not None:
        for label in labels:
            b = genes[f"pval_{label}_bonf"]
            ok = b.notnull()
            genes[f"qval_{label}"] = np.nan
            genes.loc[ok, f"qval_{label}"] = padjust_bh(b[ok].values)  # across all chromosomes
        genes.to_csv(dirs["out"] / "cosinor_top_assoc_all_chr.tsv.gz", sep="\t", float_format="%.6g")
        print("  Saved cosinor_top_assoc_all_chr.tsv.gz (q-values recomputed genome-wide)")

    # ── Summary stats ──────────────────────────────────────────────────────
    print(f"\nSummary:")
    print(f"  Total pairs tested:      {len(df):,}")
    print(f"  Bonferroni threshold:    {bonf:.2e}")
    for label in labels:
        sig = df[df[f"pval_{label}"] < bonf]
        print(f"  {label}:")
        print(f"    Significant pairs:     {len(sig):,}")
        print(f"    Significant genes:     {sig['phenotype_id'].nunique()}")
        if genes is not None:
            print(f"    Genes with q < 0.05:   {int((genes[f'qval_{label}'] < 0.05).sum())}"
                  f" (of {len(genes)}; Bonferroni within gene, BH across genes)")
    if any(t in ("3df", "5df") for t in tests):
        print("  Note: 3df/5df include the SNP main effect and are dominated by ordinary eQTLs;"
              " read them together with pval_g and 2df/4df.")
    print(f"\nPlots saved to: {dirs['plots']}/")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def make_dirs(base):
    d = {
        "base":  Path(base),
        "raw":   Path(base) / "raw",
        "data":  Path(base) / "data",
        "geno":  Path(base) / "geno",
        "out":   Path(base) / "qtl_out",
        "plots": Path(base) / "plots",
        "logs":  Path(base) / "logs",
        "tmp":   Path(base) / "tmp",
    }
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--project", required=True,
                        help="Puhti project ID (e.g. project_2012345).")
    parser.add_argument("--workdir", default="pipeline_output",
                        help="Base output directory (default: pipeline_output).")
    parser.add_argument("--step",
                        choices=["download", "chiral", "preprocess", "map", "plots"],
                        help="Run a single step.")
    parser.add_argument("--all", action="store_true",
                        help="Run all steps sequentially.")
    parser.add_argument("--chroms", default=",".join(ALL_CHROMS),
                        help="Comma-separated chromosomes to map "
                             "(default: all 22). Example: --chroms 12,17")
    parser.add_argument("--no-slurm", action="store_true",
                        help="Run mapping locally instead of submitting SLURM jobs.")
    parser.add_argument("--n-iter", type=int, default=200,
                        help="CHIRAL EM iterations (default: 200).")
    parser.add_argument("--n-top-genes", type=int, default=500,
                        help="Top variable genes for CHIRAL (default: 500, fewer = less memory).")
    parser.add_argument("--harmonics", type=int, choices=[1, 2], default=None,
                        help="Cosinor harmonics written by the preprocess step: 1 = cos_t, sin_t; "
                             "2 = also cos2_t, sin2_t (default: 2 if --tests includes 4df/5df, else 1).")
    parser.add_argument("--tests", default="2df",
                        help="Comma-separated joint tests from {2df,3df,4df,5df} (default: 2df). "
                             "2df = 2-DF (2-DOF) joint sin/cos test. The first test ranks the top hits.")
    args = parser.parse_args()

    tests = list(dict.fromkeys(t.strip() for t in args.tests.split(",") if t.strip()))
    unknown = [t for t in tests if t not in VALID_TESTS]
    if not tests or unknown:
        parser.error(f"--tests must be a comma-separated list from {VALID_TESTS}")
    needed = 2 if HARMONIC2_TESTS.intersection(tests) else 1
    if args.harmonics is None:
        args.harmonics = needed
    elif args.harmonics < needed:
        parser.error(f"--tests {args.tests} requires --harmonics 2")

    if not args.step and not args.all:
        parser.print_help()
        sys.exit(1)

    dirs   = make_dirs(args.workdir)
    chroms = [c.strip() for c in args.chroms.split(",")]

    steps = ([args.step] if args.step
             else ["download", "chiral", "preprocess", "map", "plots"])

    for step in steps:
        t0 = time.time()
        if step == "download":
            step_download(dirs)
        elif step == "chiral":
            step_chiral(dirs, n_iter=args.n_iter, n_top=args.n_top_genes)
        elif step == "preprocess":
            step_preprocess(dirs, harmonics=args.harmonics)
        elif step == "map":
            step_map(dirs, args.project, chroms=chroms,
                     use_slurm=not args.no_slurm, tests=tests)
        elif step == "plots":
            step_plots(dirs, tests=tests)
        print(f"  [{step}] done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
