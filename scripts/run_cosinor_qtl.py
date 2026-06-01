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
import numpy as np
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
        sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from tensorqtl import genotypeio
        from tensorqtl.core import read_phenotype_bed
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
    covariates_df = pd.read_csv(covariates_path, sep="\t", index_col=0).T

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

    Output columns follow standard tensorQTL format. Interaction columns
    are labeled with the interaction term name (cos_t), e.g.:
        b_g_x_cos_t, b_g_x_cos_t_se, pval_g_x_cos_t
    """
    try:
        sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from tensorqtl import cis
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
        f"Running cosinor cis-QTL mapping (1-DF interaction: SNP x cos_t)\n"
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
