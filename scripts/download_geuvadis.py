#!/usr/bin/env python3
"""Download and preprocess the full GEUVADIS dataset for cosinor eQTL analysis.

Downloads:
  - Gene expression (RPKM) for all 462 samples from EBI ArrayExpress
  - Sample metadata (population labels, sample IDs)
  - Gene annotation (Ensembl GRCh38) for building the BED file

Outputs (in --out-dir):
  - geuvadis_expression.bed.gz       tensorQTL-format BED (log-norm, inv-normal)
  - geuvadis_covariates.txt          PCs for eQTL mapping (from expression PCA)
  - geuvadis_samples_EUR.txt         EUR sample IDs (for genotype filtering)
  - geuvadis_raw_rpkm.tsv.gz        Raw RPKM matrix (input to CHIRAL)

NOTE: GEUVADIS uses LCL (lymphoblastoid cell lines) which have weak circadian
signal. CHIRAL will run but inferred phases should be treated as exploratory.
For reliable circadian eQTL results, use a tissue with strong rhythmic expression
(blood with timed draws, liver, etc.).

Genotype data (not downloaded here — large files):
  1000 Genomes Phase 3 VCFs are at:
  http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/
  After download, convert with:
    bcftools view -S geuvadis_samples_EUR.txt chr1.vcf.gz | \\
    plink2 --vcf /dev/stdin --make-pgen --maf 0.01 --out chr1
"""
import argparse
import gzip
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy import stats

EBI_BASE = "http://ftp.ebi.ac.uk/pub/databases/microarray/data/experiment/GEUV/E-GEUV-1"
RPKM_URL = f"{EBI_BASE}/analysis_results/GD462.GeneQuantRPKM.50FN.samplename.resk10.txt.gz"
SDRF_URL = "http://www.ebi.ac.uk/arrayexpress/files/E-GEUV-1/E-GEUV-1.sdrf.txt"

# Ensembl GRCh38 gene annotation (genes only, no patches)
GTF_URL = (
    "https://ftp.ensembl.org/pub/release-111/gtf/homo_sapiens/"
    "Homo_sapiens.GRCh38.111.chr.gtf.gz"
)


def download(url: str, dest: str) -> None:
    if os.path.exists(dest):
        print(f"  Already exists: {dest}")
        return
    print(f"  Downloading {url}")
    subprocess.run(["wget", "-q", "-O", dest, url], check=True)


def load_rpkm(path: str) -> pd.DataFrame:
    """Load GD462 RPKM matrix. Returns genes x samples DataFrame."""
    print("Loading RPKM matrix...")
    df = pd.read_csv(path, sep="\t", index_col=0)
    # Drop non-sample metadata columns if present
    sample_cols = [c for c in df.columns if not c.startswith("Gene")]
    return df[sample_cols]


def load_sample_metadata(path: str) -> pd.DataFrame:
    """Load SDRF metadata and extract population + sample IDs."""
    df = pd.read_csv(path, sep="\t")
    # Keep one row per sample
    df = df.drop_duplicates(subset=["Source Name"])
    df = df[["Source Name", "Characteristics[ancestry category]"]].copy()
    df.columns = ["sample_id", "population"]
    return df.set_index("sample_id")


def get_eur_samples(rpkm: pd.DataFrame, metadata: pd.DataFrame) -> list:
    """Return EUR sample IDs present in both RPKM and metadata."""
    eur = metadata[metadata["population"].isin(
        ["British", "Finnish", "Utah", "Tuscan"]
    )].index
    return [s for s in rpkm.columns if s in eur]


def inverse_normal_transform(x: np.ndarray) -> np.ndarray:
    """Rank-based inverse normal transform for a 1-D array."""
    ranks = stats.rankdata(x, method="average")
    return stats.norm.ppf(ranks / (len(ranks) + 1))


def normalize_expression(rpkm: pd.DataFrame, min_samples_expressed: float = 0.1) -> pd.DataFrame:
    """Log-normalize and inverse-normal transform RPKM for eQTL mapping.

    Steps:
      1. Filter genes expressed (RPKM > 0.1) in >= min_samples_expressed fraction
      2. Log2(RPKM + 1) transform
      3. Inverse normal transform per gene across samples
    """
    print("Normalizing expression...")
    n_samples = rpkm.shape[1]
    expressed = (rpkm > 0.1).sum(axis=1) >= int(min_samples_expressed * n_samples)
    rpkm = rpkm.loc[expressed]
    print(f"  Kept {rpkm.shape[0]} genes expressed in ≥{min_samples_expressed*100:.0f}% of samples")

    # Clip negative values before log transform (some processed RPKM files have small negatives)
    rpkm = rpkm.clip(lower=0)
    log_rpkm = np.log2(rpkm + 1)

    # Drop any genes/samples that still have NaN or inf after log transform
    log_rpkm = log_rpkm.replace([np.inf, -np.inf], np.nan)
    log_rpkm = log_rpkm.dropna(how="any")
    print(f"  {log_rpkm.shape[0]} genes after NaN/inf filter")

    normed = log_rpkm.apply(lambda row: inverse_normal_transform(row.values), axis=1,
                            result_type="expand")
    normed.columns = log_rpkm.columns
    normed.index = log_rpkm.index
    return normed


def build_bed(normed: pd.DataFrame, gtf_path: str, out_path: str) -> None:
    """Build tensorQTL-format BED file by adding genomic coordinates from GTF."""
    print("Building BED file from GTF annotation...")

    coords = {}
    with gzip.open(gtf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            if fields[2] != "gene":
                continue
            chrom = fields[0]
            start = int(fields[3]) - 1   # 0-based
            end = int(fields[4])
            attrs = fields[8]
            gene_id = None
            for attr in attrs.split(";"):
                attr = attr.strip()
                if attr.startswith("gene_id"):
                    gene_id = attr.split('"')[1].split(".")[0]
                    break
            if gene_id:
                coords[gene_id] = (chrom, start, end)

    rows = []
    for gene in normed.index:
        base_id = gene.split(".")[0]
        if base_id in coords:
            chrom, start, end = coords[base_id]
            if not chrom.startswith("CHR") and "_" not in chrom:
                # Add chr prefix to match UCSC/PLINK format (Ensembl uses bare numbers)
                chrom = chrom if chrom.startswith("chr") else f"chr{chrom}"
                rows.append((chrom, start, end, gene))

    coord_df = pd.DataFrame(rows, columns=["#chr", "start", "end", "gene_id"])
    coord_df = coord_df.sort_values(["#chr", "start"])

    normed_reindexed = normed.reindex(coord_df["gene_id"])
    bed = pd.concat([coord_df.set_index("gene_id"), normed_reindexed], axis=1)
    bed.index.name = "gene_id"
    bed = bed.reset_index()
    bed = bed[["#chr", "start", "end", "gene_id"] + list(normed.columns)]

    # Sort within each chromosome by start position (required by tensorQTL).
    # Use natural chromosome order so chr1 < chr2 < ... < chr22 < chrX < chrY.
    chrom_rank = {str(i): i for i in range(1, 23)}
    chrom_rank.update({"X": 23, "Y": 24, "MT": 25})
    bed["_rank"] = bed["#chr"].map(lambda c: chrom_rank.get(c, 99))
    bed = bed.sort_values(["_rank", "start", "end"]).drop(columns=["_rank"])
    bed = bed.reset_index(drop=True)

    bed.to_csv(out_path, sep="\t", index=False, compression="gzip")
    print(f"  Wrote BED: {out_path} ({len(bed)} genes)")


def build_covariates(normed: pd.DataFrame, n_pcs: int, out_path: str) -> None:
    """Compute expression PCs as covariates (proxy for PEER factors)."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    print(f"Computing {n_pcs} expression PCs as covariates...")
    X = StandardScaler().fit_transform(normed.values.T)  # samples x genes
    pcs = PCA(n_components=n_pcs).fit_transform(X)
    pc_df = pd.DataFrame(
        pcs.T,
        index=[f"ExprPC{i+1}" for i in range(n_pcs)],
        columns=normed.columns,
    )
    pc_df.to_csv(out_path, sep="\t")
    print(f"  Wrote covariates: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True,
                        help="Output directory for all downloaded and processed files.")
    parser.add_argument("--n-pcs", type=int, default=20,
                        help="Number of expression PCs to use as covariates (default: 20).")
    parser.add_argument("--eur-only", action="store_true", default=True,
                        help="Keep only EUR samples (default: True).")
    parser.add_argument("--skip-gtf", action="store_true",
                        help="Skip GTF download and BED construction (faster, no BED output).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Step 1: Download expression
    rpkm_path = os.path.join(args.out_dir, "GD462.GeneQuantRPKM.50FN.samplename.resk10.txt.gz")
    sdrf_path = os.path.join(args.out_dir, "E-GEUV-1.sdrf.txt")
    print("=== Downloading GEUVADIS expression ===")
    download(RPKM_URL, rpkm_path)
    download(SDRF_URL, sdrf_path)

    # Step 2: Load and filter samples
    rpkm = load_rpkm(rpkm_path)
    metadata = load_sample_metadata(sdrf_path)

    if args.eur_only:
        eur_samples = get_eur_samples(rpkm, metadata)
        print(f"Keeping {len(eur_samples)} EUR samples")
        rpkm = rpkm[eur_samples]
        pd.Series(eur_samples).to_csv(
            os.path.join(args.out_dir, "geuvadis_samples_EUR.txt"),
            index=False, header=False
        )

    # Step 3: Save raw RPKM for CHIRAL
    raw_path = os.path.join(args.out_dir, "geuvadis_raw_rpkm.tsv.gz")
    rpkm.to_csv(raw_path, sep="\t", compression="gzip")
    print(f"Saved raw RPKM: {raw_path}  (input for CHIRAL)")

    # Step 4: Normalize for eQTL mapping
    normed = normalize_expression(rpkm)

    # Step 5: Build BED (needs GTF)
    if not args.skip_gtf:
        gtf_path = os.path.join(args.out_dir, "Homo_sapiens.GRCh38.111.chr.gtf.gz")
        print("=== Downloading GTF annotation (~50 MB) ===")
        download(GTF_URL, gtf_path)
        bed_path = os.path.join(args.out_dir, "geuvadis_expression.bed.gz")
        build_bed(normed, gtf_path, bed_path)
    else:
        print("Skipping BED construction (--skip-gtf). Run without --skip-gtf for full pipeline.")

    # Step 6: Covariates
    cov_path = os.path.join(args.out_dir, "geuvadis_covariates.txt")
    build_covariates(normed, args.n_pcs, cov_path)

    print()
    print("=== Done ===")
    print(f"Output directory: {args.out_dir}")
    print()
    print("Next steps:")
    print(f"  1. Infer circadian phases:")
    print(f"     python scripts/infer_phases_chiral.py --rpkm {raw_path} --out-dir {args.out_dir}")
    print(f"  2. Download genotypes (large — see script docstring for instructions)")
    print(f"  3. Run cosinor_preprocess.py using the inferred phases as collection times")


if __name__ == "__main__":
    main()
