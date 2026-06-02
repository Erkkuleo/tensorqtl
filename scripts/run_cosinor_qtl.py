#!/usr/bin/env python3
"""tensorQTL runner for cosinor (circadian) cis-eQTL interaction mapping.

Runs a 1-DF interaction test: does a SNP's effect on gene expression
change with the cosine of time-of-day?

Model fitted per gene:
    expression ~ SNP + cos_t + sin_t + SNP*cos_t + other_covariates

cos_t and sin_t enter as regular covariates (via --covariates from
cosinor_preprocess.py). cos_t additionally enters as the interaction
term (via --interaction). This is a 1-DF test; for the 2-DF upgrade
see docs/cosinor_eqtl.md.
"""
import argparse
import os
import sys
import pandas as pd


def validate_sample_alignment(
    phenotype_df: pd.DataFrame,
    covariates_df: pd.DataFrame,
    interaction_df: pd.DataFrame,
) -> None:
    """Verify that all three inputs share the same samples in the same order.

    Args:
        phenotype_df: genes x samples (samples are columns)
        covariates_df: samples x covariates (samples are index)
        interaction_df: samples x interactions (samples are index)

    Raises:
        ValueError: If samples are mismatched or in a different order.
    """
    pheno_samples = phenotype_df.columns

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
        interaction_path: Interaction file (rows=samples, col header=cos_t).

    Returns:
        (genotype_df, variant_df, phenotype_df, phenotype_pos_df,
         covariates_df, interaction_df)

        covariates_df: samples x covariates (transposed from file)
        interaction_df: samples x 1, indexed by sample IDs
    """
    try:
        _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _pkg_dir = os.path.join(_repo_root, "tensorqtl")
        # Insert the package directory so submodules can be imported directly
        # (without triggering tensorqtl/__init__.py which imports optional deps).
        for _p in (_pkg_dir, _repo_root):
            if _p not in sys.path:
                sys.path.insert(1, _p)
        import genotypeio  # type: ignore[import]
        from core import read_phenotype_bed  # type: ignore[import]
    except ImportError as e:
        raise ImportError(
            "tensorqtl is not installed. Install it or run from the repo root."
        ) from e

    print(f"Loading phenotypes from {phenotypes_path}")
    phenotype_df, phenotype_pos_df = read_phenotype_bed(phenotypes_path)

    print(f"Loading genotypes from {plink_prefix}")
    genotype_df, variant_df = genotypeio.load_genotypes(
        plink_prefix, select_samples=phenotype_df.columns
    )

    print(f"Loading covariates from {covariates_path}")
    # Read header manually to avoid a pandas 2.x crash (TypeError in
    # maybe_convert_objects) that affects files with many columns whose
    # names trigger type inference (e.g. "NA06984"-style sample IDs).
    with open(covariates_path) as _f:
        _sample_ids = _f.readline().rstrip("\n").split("\t")[1:]
    _cov = pd.read_csv(covariates_path, sep="\t", skiprows=1, header=None, index_col=0)
    _cov.columns = _sample_ids
    covariates_df = _cov.T

    print(f"Loading interaction terms from {interaction_path}")
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

    Output columns follow standard tensorQTL format. For a single
    interaction term named 'cos_t', the columns are:
        pval_g, b_g, b_g_se          – main SNP effect
        pval_cos_t, b_cos_t          – main interaction (cos_t) effect
        pval_g_x_cos_t, b_g_x_cos_t – SNP x cos_t interaction effect
    """
    try:
        _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _pkg_dir = os.path.join(_repo_root, "tensorqtl")
        for _p in (_pkg_dir, _repo_root):
            if _p not in sys.path:
                sys.path.insert(1, _p)
        import cis  # type: ignore[import]
    except ImportError as e:
        raise ImportError(
            "tensorqtl is not installed. Install it or run from the repo root."
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


def compute_2df_pvalues(output_dir: str, prefix: str) -> None:
    """Add a joint 2-DF cosinor p-value column to each output parquet.

    Combines the SNP×cos_t and SNP×sin_t t-statistics into a single
    chi-squared statistic with 2 degrees of freedom:

        chi2 = t_cos² + t_sin²   (valid when cos_t and sin_t are uncorrelated)
        pval_g_x_cosinor_2df = P(Chi2(2) > chi2)

    This tests H0: β_cos_t = β_sin_t = 0 simultaneously and is sensitive
    to circadian interactions at any phase. Overwrites each parquet in-place.
    """
    from scipy import stats as scipy_stats

    parquets = sorted([
        f for f in os.listdir(output_dir)
        if f.startswith(prefix) and f.endswith(".parquet")
    ])
    if not parquets:
        raise FileNotFoundError(
            f"No parquet files found in {output_dir} with prefix '{prefix}'"
        )
    for fname in parquets:
        path = os.path.join(output_dir, fname)
        df = pd.read_parquet(path)
        t_cos = df["b_g_x_cos_t"] / df["b_g_x_cos_t_se"]
        t_sin = df["b_g_x_sin_t"] / df["b_g_x_sin_t_se"]
        chi2 = t_cos**2 + t_sin**2
        df["pval_g_x_cosinor_2df"] = scipy_stats.chi2.sf(chi2.values, df=2)
        df.to_parquet(path)
        print(f"  Added pval_g_x_cosinor_2df to {fname}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run cosinor cis-eQTL interaction mapping using tensorQTL. "
            "Tests whether each SNP's effect on expression varies with time of day. "
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
                        help="Interaction file output by cosinor_preprocess.py (cos_t + sin_t columns).")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write result parquet files.")
    parser.add_argument("--prefix", required=True,
                        help="Output filename prefix (e.g. 'my_study.cosinor').")
    parser.add_argument("--window", type=int, default=1_000_000,
                        help="cis window size in bp (default: 1000000).")
    parser.add_argument("--cosinor-2df", action="store_true",
                        help="After mapping, compute a joint 2-DF cosinor p-value "
                             "(pval_g_x_cosinor_2df) combining SNP×cos_t and SNP×sin_t. "
                             "Requires the interaction file to have both cos_t and sin_t columns.")
    args = parser.parse_args()

    (genotype_df, variant_df, phenotype_df, phenotype_pos_df,
     covariates_df, interaction_df) = load_inputs(
        args.plink_prefix, args.phenotypes, args.covariates, args.interaction
    )

    if args.cosinor_2df:
        missing = [c for c in ("cos_t", "sin_t") if c not in interaction_df.columns]
        if missing:
            raise ValueError(
                f"--cosinor-2df requires both 'cos_t' and 'sin_t' columns in the "
                f"interaction file, but missing: {missing}"
            )

    validate_sample_alignment(phenotype_df, covariates_df, interaction_df)

    mode = "2-DF joint (SNP × cos_t + SNP × sin_t)" if args.cosinor_2df else "per interaction term"
    print(
        f"Running cosinor cis-QTL mapping [{mode}]\n"
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

    if args.cosinor_2df:
        print("Computing joint 2-DF cosinor p-values...")
        compute_2df_pvalues(args.output_dir, args.prefix)

    print(f"Done. Results: {args.output_dir}/{args.prefix}.cis_qtl_pairs.*.parquet")


if __name__ == "__main__":
    main()
