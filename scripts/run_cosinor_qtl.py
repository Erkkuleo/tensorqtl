#!/usr/bin/env python3
"""tensorQTL runner for cosinor (circadian) cis-eQTL interaction mapping.

Model fitted per gene and cis variant (covariates partialled out):

    harmonics = 1:  y ~ g + c1 + s1 + g:c1 + g:s1 + covariates
    harmonics = 2:  y ~ g + c1 + s1 + c2 + s2 + g:c1 + g:s1 + g:c2 + g:s2 + covariates

with c1, s1 = cos_t, sin_t (24 h) and c2, s2 = cos2_t, sin2_t (12 h), taken
from the --interaction file written by cosinor_preprocess.py. tensorQTL enters
each interaction column as a main effect, so these terms must not also be in
the covariates.

Joint Wald F-tests (--tests, comma-separated):

    2df  g:c1, g:s1                 2-DF (2-DOF) joint sin/cos test: genotype
                                    changes 24 h amplitude and/or phase
    3df  g, g:c1, g:s1              mesor shift and/or 24 h rhythm change
    4df  g:c1, g:s1, g:c2, g:s2     rhythm shape change at 24 h or 12 h
    5df  g, g:c1 ... g:s2           any genotype effect (dominated by ordinary eQTLs)

2df/3df alone are computed from the harmonics = 1 model. If 4df or 5df is
requested, all tests come from the single harmonics = 2 fit; the output
column names record this (pval_cosinor_2df_h1 vs pval_cosinor_2df_h2).
The top association per gene is chosen by the first test in --tests.
See docs/cosinor_eqtl.md.
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


def _tensorqtl_path() -> None:
    """Make tensorqtl submodules importable directly (without tensorqtl/__init__.py,
    which imports optional dependencies)."""
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _pkg_dir = os.path.join(_repo_root, "tensorqtl")
    for _p in (_pkg_dir, _repo_root):
        if _p not in sys.path:
            sys.path.insert(1, _p)


def check_no_overlap(covariates_df: pd.DataFrame, interaction_df: pd.DataFrame) -> None:
    """Raise if an interaction term is also a covariate (singular design matrix).

    Args:
        covariates_df: samples x covariates
        interaction_df: samples x interactions
    """
    overlap = [c for c in interaction_df.columns if c in covariates_df.columns]
    if overlap:
        raise ValueError(
            f"Interaction term(s) {overlap} also present in the covariates. "
            "tensorQTL already includes interaction terms as main effects; "
            "remove them from the covariates file."
        )


def parse_tests(value: str) -> list:
    """Parse a comma-separated --tests value, e.g. '2df,4df'."""
    _tensorqtl_path()
    import cosinor  # type: ignore[import]
    tests = [t.strip() for t in value.split(",") if t.strip()]
    if not tests:
        raise ValueError("--tests must name at least one test")
    unknown = [t for t in tests if t not in cosinor.COSINOR_TESTS]
    if unknown:
        raise ValueError(f"Unknown test(s) {unknown}; choose from {list(cosinor.COSINOR_TESTS)}")
    return list(dict.fromkeys(tests))  # drop duplicates, keep order


def prepare_interaction(interaction_df: pd.DataFrame, tests: list) -> pd.DataFrame:
    """Select the cosinor columns of the model needed for the requested tests.

    2df/3df only -> harmonics = 1 model (cos_t, sin_t), even if the file also
    has cos2_t/sin2_t; any of 4df/5df -> harmonics = 2 model.

    Raises:
        ValueError: if the interaction file lacks the required columns.
    """
    _tensorqtl_path()
    import cosinor  # type: ignore[import]
    harmonics = cosinor.model_harmonics(tests)
    required = cosinor.HARMONIC_COLUMNS[harmonics]
    missing = [c for c in required if c not in interaction_df.columns]
    if missing:
        need = [t for t in tests if cosinor.COSINOR_TESTS[t]["harmonics"] == harmonics]
        hint = (" Re-run cosinor_preprocess.py with --harmonics 2." if harmonics == 2 else "")
        raise ValueError(
            f"--tests {','.join(need)} requires interaction columns {required}, "
            f"but the interaction file is missing {missing}.{hint}"
        )
    extra = [c for c in interaction_df.columns if c not in required]
    if extra:
        print(f"  * using the harmonics = {harmonics} model; ignoring interaction columns {extra}")
    return interaction_df[required]


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
        interaction_path: Interaction file (rows=samples, cols=cos_t, sin_t[, cos2_t, sin2_t]).

    Returns:
        (genotype_df, variant_df, phenotype_df, phenotype_pos_df,
         covariates_df, interaction_df)

        covariates_df: samples x covariates (transposed from file)
        interaction_df: samples x interactions, indexed by sample IDs
    """
    try:
        _tensorqtl_path()
        import genotypeio  # type: ignore[import]
        from core import read_phenotype_bed  # type: ignore[import]
    except ImportError as e:
        raise ImportError(
            "tensorqtl is not installed. Install it or run from the repo root."
        ) from e

    print(f"Loading phenotypes from {phenotypes_path}")
    phenotype_df, phenotype_pos_df = read_phenotype_bed(phenotypes_path)

    # Read genotype sample list from psam to find intersection with phenotype samples
    psam_path = plink_prefix + ".psam"
    if os.path.exists(psam_path):
        psam = pd.read_csv(psam_path, sep="\t", usecols=[0])
        geno_samples = set(psam.iloc[:, 0].astype(str))
        shared = [s for s in phenotype_df.columns if s in geno_samples]
        if len(shared) < len(phenotype_df.columns):
            print(f"  * {len(phenotype_df.columns) - len(shared)} phenotype samples "
                  f"not in genotype file — using {len(shared)} overlapping samples")
        phenotype_df = phenotype_df[shared]
    else:
        shared = list(phenotype_df.columns)

    print(f"Loading genotypes from {plink_prefix}")
    genotype_df, variant_df = genotypeio.load_genotypes(
        plink_prefix, select_samples=shared
    )

    print(f"Loading covariates from {covariates_path}")
    # Read header manually to avoid a pandas 2.x crash (TypeError in
    # maybe_convert_objects) that affects files with many columns whose
    # names trigger type inference (e.g. "NA06984"-style sample IDs).
    with open(covariates_path) as _f:
        _sample_ids = _f.readline().rstrip("\n").split("\t")[1:]
    _cov = pd.read_csv(covariates_path, sep="\t", skiprows=1, header=None, index_col=0)
    _cov.columns = _sample_ids
    covariates_df = _cov.T.reindex(shared)  # subset to overlapping samples

    print(f"Loading interaction terms from {interaction_path}")
    interaction_df = pd.read_csv(interaction_path, sep="\t", index_col=0).reindex(shared)

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
    tests: list = ("2df",),
) -> None:
    """Run tensorQTL cis nominal mapping with cosinor interaction terms.

    Writes per-chromosome results to:
        <output_dir>/<prefix>.cis_qtl_pairs.<chr>.parquet
    and the top association per gene (by the first test in `tests`) to:
        <output_dir>/<prefix>.cis_qtl_top_assoc.txt.gz

    Pair columns, besides the standard tensorQTL per-term columns
    (pval_g, pval_cos_t, pval_g_x_cos_t, b_g_x_cos_t, ...):
        F_cosinor_<test>_h<k>, pval_cosinor_<test>_h<k>
    Top-association columns additionally include num_var,
    pval_cosinor_<test>_h<k>_bonf and qval_cosinor_<test>_h<k>.
    """
    try:
        _tensorqtl_path()
        import cis  # type: ignore[import]
        import cosinor  # type: ignore[import]
    except ImportError as e:
        raise ImportError(
            "tensorqtl is not installed. Install it or run from the repo root."
        ) from e

    joint_tests = cosinor.build_joint_tests(list(tests), interaction_df.columns)
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
        joint_tests=joint_tests,
        top_by=next(iter(joint_tests)),
    )


def _acat(p1: "pd.Series", p2: "pd.Series") -> "pd.Series":
    """Aggregated Cauchy Association Test for two correlated p-values.

    Valid even when p1 and p2 are correlated (unlike chi-squared sum of
    t-statistics, which assumes independence). Under the null, the ACAT
    statistic follows a standard Cauchy distribution regardless of the
    correlation between the input tests.

    Reference: Liu & Xie (2020) ACAT: A Fast and Powerful p-value
    Combination Method for Rare-Variant Analysis. Am J Hum Genet.
    """
    import numpy as np
    from scipy import stats as scipy_stats
    p1 = p1.clip(lower=1e-300, upper=1 - 1e-15)
    p2 = p2.clip(lower=1e-300, upper=1 - 1e-15)
    T = 0.5 * (np.tan((0.5 - p1) * np.pi) + np.tan((0.5 - p2) * np.pi))
    return pd.Series(0.5 - np.arctan(T.values) / np.pi, index=p1.index)


def compute_acat_pvalues(output_dir: str, prefix: str) -> None:
    """Add pval_g_x_cosinor_acat to each output parquet (comparison only).

    ACAT combination of pval_g_x_cos_t and pval_g_x_sin_t. This was the
    previous "2-DF" p-value; it is not a joint test of both coefficients.
    Use pval_cosinor_2df_h<k> (joint Wald F-test) for inference.
    Overwrites each parquet in-place with the new column appended.
    """
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
        df["pval_g_x_cosinor_acat"] = _acat(
            df["pval_g_x_cos_t"], df["pval_g_x_sin_t"]
        )
        df.to_parquet(path)
        print(f"  Added pval_g_x_cosinor_acat to {fname}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run cosinor cis-eQTL interaction mapping using tensorQTL. "
            "Tests whether each SNP's effect on expression varies with time of day "
            "using joint Wald F-tests of the genotype x cosinor terms. "
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
                        help="Interaction file output by cosinor_preprocess.py "
                             "(cos_t, sin_t; plus cos2_t, sin2_t with --harmonics 2).")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write result parquet files.")
    parser.add_argument("--prefix", required=True,
                        help="Output filename prefix (e.g. 'my_study.cosinor').")
    parser.add_argument("--window", type=int, default=1_000_000,
                        help="cis window size in bp (default: 1000000).")
    parser.add_argument("--tests", default=None,
                        help="Comma-separated joint tests from {2df,3df,4df,5df} (default: 2df). "
                             "2df = 2-DF (2-DOF) joint sin/cos test of SNP x cos_t and SNP x sin_t; "
                             "3df adds the SNP main effect; 4df = SNP x 24 h and 12 h terms "
                             "(needs cos2_t/sin2_t); 5df = 4df + SNP main effect. "
                             "The first test selects the top variant per gene.")
    parser.add_argument("--also-acat", action="store_true",
                        help="Also write pval_g_x_cosinor_acat (ACAT of the two 1-DF p-values, "
                             "the previous approximation) for comparison.")
    parser.add_argument("--cosinor-2df", action="store_true",
                        help="Deprecated alias for --tests 2df.")
    return parser


def resolve_tests(args: argparse.Namespace) -> list:
    """Tests to run, handling the deprecated --cosinor-2df alias."""
    if args.cosinor_2df:
        print("WARNING: --cosinor-2df is deprecated; use --tests 2df. "
              "The joint 2-DF p-value is now a Wald F-test (pval_cosinor_2df_h<k>), not ACAT.",
              file=sys.stderr)
        if args.tests is None:
            return ["2df"]
        tests = parse_tests(args.tests)
        return tests if "2df" in tests else ["2df"] + tests
    return parse_tests(args.tests if args.tests is not None else "2df")


def main() -> None:
    args = build_parser().parse_args()
    tests = resolve_tests(args)

    (genotype_df, variant_df, phenotype_df, phenotype_pos_df,
     covariates_df, interaction_df) = load_inputs(
        args.plink_prefix, args.phenotypes, args.covariates, args.interaction
    )

    interaction_df = prepare_interaction(interaction_df, tests)
    validate_sample_alignment(phenotype_df, covariates_df, interaction_df)
    check_no_overlap(covariates_df, interaction_df)

    print(
        f"Running cosinor cis-QTL mapping [joint tests: {', '.join(tests)}]\n"
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
        tests=tests,
    )

    if args.also_acat:
        print("Computing ACAT p-values for comparison...")
        compute_acat_pvalues(args.output_dir, args.prefix)

    print(f"Done. Results: {args.output_dir}/{args.prefix}.cis_qtl_pairs.*.parquet")


if __name__ == "__main__":
    main()
