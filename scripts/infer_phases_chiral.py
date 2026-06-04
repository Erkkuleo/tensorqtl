#!/usr/bin/env python3
"""Infer circadian phases from bulk RNA-seq using CHIRAL (Naef lab, EPFL).

CHIRAL (Circular Harmonic Inference from RNA-seq ALgorithm) estimates the
circadian phase of each sample from a gene expression matrix using an
expectation-maximisation algorithm over a cosinor model.

Reference: Talamanca & Gobet et al., Naef Lab, EPFL.
           https://github.com/naef-lab/CHIRAL

Input:
  - Raw RPKM expression matrix (genes x samples, TSV format)
    Produced by download_geuvadis.py as geuvadis_raw_rpkm.tsv.gz

Output:
  - chiral_phases.tsv    Sample IDs + inferred hour (0-24) + phase (radians)
                         Ready for use as --metadata in cosinor_preprocess.py

Usage:
  python scripts/infer_phases_chiral.py \\
      --rpkm data/geuvadis_raw_rpkm.tsv.gz \\
      --out-dir data/

Then:
  python scripts/cosinor_preprocess.py \\
      --metadata data/chiral_phases.tsv \\
      --covariates data/geuvadis_covariates.txt \\
      --out-covariates data/cov_cosinor.txt \\
      --out-interaction data/interaction.txt \\
      --time-col hour \\
      --mode tod

NOTE: GEUVADIS uses LCLs (cell lines) with weak circadian signal. CHIRAL
will run but the inferred phases reflect residual expression variation rather
than true circadian timing. This script serves as a template — apply it to
tissues with strong rhythmic expression (blood, liver, lung) for meaningful
circadian eQTL analysis.
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd


CHIRAL_R_SCRIPT = r"""
# Install CHIRAL from GitHub if not already installed
if (!requireNamespace("CHIRAL", quietly = TRUE)) {
  if (!requireNamespace("remotes", quietly = TRUE)) {
    install.packages("remotes", repos = "http://cran.us.r-project.org", quiet = TRUE)
  }
  remotes::install_github("naef-lab/CHIRAL/Pkg/CHIRAL", quiet = TRUE)
}

library(CHIRAL)

args <- commandArgs(trailingOnly = TRUE)
rpkm_path <- args[1]
out_path   <- args[2]
n_iter     <- as.integer(args[3])
gtex_names <- as.logical(args[4])

cat(sprintf("Loading expression matrix from %s\n", rpkm_path))
E <- as.matrix(read.table(rpkm_path, sep = "\t", header = TRUE, row.names = 1,
                           check.names = FALSE))

cat(sprintf("Expression matrix: %d genes x %d samples\n", nrow(E), ncol(E)))

# Log2 transform RPKM (CHIRAL expects log-scale input)
E <- log2(E + 1)

cat(sprintf("Running CHIRAL (iterations = %d)...\n", n_iter))
result <- CHIRAL(
  E             = E,
  iterations    = n_iter,
  GTEx_names    = gtex_names,
  mean.centre.E = TRUE,
  TSM           = TRUE,
  pbar          = TRUE
)

# Extract phases (radians, 0 to 2*pi) and convert to hours (0 to 24)
phi   <- result$phi
hours <- phi * 24 / (2 * pi)

# Write output TSV
out_df <- data.frame(
  sample_id     = names(phi),
  hour          = round(hours, 4),
  phase_radians = round(phi, 6),
  row.names     = NULL
)
write.table(out_df, out_path, sep = "\t", row.names = FALSE, quote = FALSE)
cat(sprintf("Wrote phase estimates to %s\n", out_path))
cat(sprintf("Phase range: %.2f - %.2f hours\n", min(hours), max(hours)))
"""


def check_r_available() -> str:
    """Return the path to Rscript, or raise if not found."""
    for cmd in ["Rscript", "Rscript.exe"]:
        try:
            result = subprocess.run([cmd, "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                return cmd
        except FileNotFoundError:
            continue
    raise RuntimeError(
        "Rscript not found. Install R first:\n"
        "  Ubuntu/Colab: sudo apt-get install r-base\n"
        "  conda: conda install -c conda-forge r-base"
    )


def prepare_rpkm_for_r(rpkm_path: str, tmp_dir: str, n_top_genes: int = 5000) -> str:
    """Load RPKM, select top variable genes, write a temp TSV for R.

    CHIRAL is faster and more accurate with highly variable genes.
    Using all ~20k genes is slow; 2000-5000 highly variable genes works well.
    """
    print(f"Loading expression from {rpkm_path}...")
    expr = pd.read_csv(rpkm_path, sep="\t", index_col=0, compression="infer")
    print(f"  {expr.shape[0]} genes x {expr.shape[1]} samples")

    # Filter lowly expressed genes (RPKM < 1 in >80% of samples)
    expressed = (expr > 1).sum(axis=1) >= int(0.2 * expr.shape[1])
    expr = expr.loc[expressed]
    print(f"  {expr.shape[0]} genes after low-expression filter")

    # Select top variable genes by CV (coefficient of variation)
    log_expr = np.log2(expr + 1)
    cv = log_expr.std(axis=1) / (log_expr.mean(axis=1) + 1e-6)
    top_genes = cv.nlargest(min(n_top_genes, len(cv))).index
    expr_top = expr.loc[top_genes]
    print(f"  Using top {len(top_genes)} variable genes for phase inference")

    tmp_path = os.path.join(tmp_dir, "rpkm_for_chiral.tsv")
    expr_top.to_csv(tmp_path, sep="\t")
    return tmp_path


def run_chiral(rpkm_r_path: str, out_path: str, n_iter: int, gtex_names: bool,
               rscript: str) -> None:
    """Write and execute the CHIRAL R script."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False) as f:
        f.write(CHIRAL_R_SCRIPT)
        r_script_path = f.name

    try:
        cmd = [
            rscript, r_script_path,
            rpkm_r_path,
            out_path,
            str(n_iter),
            str(gtex_names).upper(),
        ]
        print(f"Running: {' '.join(cmd[:3])} ...")
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"CHIRAL R script failed with exit code {result.returncode}")
    finally:
        os.unlink(r_script_path)


def plot_phase_distribution(phases_path: str, out_dir: str) -> None:
    """Plot histogram of inferred phases as a sanity check."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    df = pd.read_csv(phases_path, sep="\t")
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.hist(df["hour"], bins=24, range=(0, 24), color="steelblue",
            edgecolor="white", alpha=0.8)
    ax.set_xlabel("Inferred hour (0-24)")
    ax.set_ylabel("Number of samples")
    ax.set_title("CHIRAL — inferred circadian phase distribution")
    ax.set_xticks(range(0, 25, 4))

    plot_path = os.path.join(out_dir, "chiral_phase_distribution.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved phase distribution plot: {plot_path}")

    if df["hour"].std() < 1.0:
        print("WARNING: Phase distribution is very narrow (SD < 1 h).")
        print("         This may indicate weak circadian signal in this tissue.")
        print("         Consider using a tissue with stronger rhythmic expression.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rpkm", required=True,
                        help="Raw RPKM expression matrix (genes x samples, TSV or TSV.gz). "
                             "Output of download_geuvadis.py.")
    parser.add_argument("--out-dir", required=True,
                        help="Output directory. Writes chiral_phases.tsv here.")
    parser.add_argument("--n-iter", type=int, default=500,
                        help="CHIRAL EM iterations (default: 500). "
                             "Reduce to 100 for a quick test.")
    parser.add_argument("--n-top-genes", type=int, default=3000,
                        help="Number of top variable genes for inference (default: 3000). "
                             "More genes = slower but potentially more accurate.")
    parser.add_argument("--gtex-names", action="store_true",
                        help="Use GTEx gene name conventions (ENSG without version suffix). "
                             "Enable when using GTEx expression data.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_phases = os.path.join(args.out_dir, "chiral_phases.tsv")

    # Check R is available
    rscript = check_r_available()
    print(f"Using R: {rscript}")

    # Prepare expression matrix
    with tempfile.TemporaryDirectory() as tmp_dir:
        rpkm_r_path = prepare_rpkm_for_r(args.rpkm, tmp_dir, args.n_top_genes)

        # Run CHIRAL
        run_chiral(rpkm_r_path, out_phases, args.n_iter, args.gtex_names, rscript)

    # Verify output
    if not os.path.exists(out_phases):
        sys.exit("ERROR: CHIRAL did not produce output. Check R error messages above.")

    phases = pd.read_csv(out_phases, sep="\t")
    print(f"\nPhase inference complete:")
    print(f"  Samples:       {len(phases)}")
    print(f"  Hour range:    {phases['hour'].min():.2f} - {phases['hour'].max():.2f}")
    print(f"  Hour SD:       {phases['hour'].std():.2f}  (expect ~6.9 for uniform)")
    print(f"  Output:        {out_phases}")

    # Sanity check plot
    plot_phase_distribution(out_phases, args.out_dir)

    print()
    print("Next step — run cosinor preprocessing with inferred phases:")
    print(f"  python scripts/cosinor_preprocess.py \\")
    print(f"      --metadata {out_phases} \\")
    print(f"      --covariates {args.out_dir}/geuvadis_covariates.txt \\")
    print(f"      --out-covariates {args.out_dir}/cov_cosinor.txt \\")
    print(f"      --out-interaction {args.out_dir}/interaction.txt \\")
    print(f"      --time-col hour --mode tod")


if __name__ == "__main__":
    main()
