import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from run_cosinor_qtl import (
    validate_sample_alignment, check_no_overlap, build_parser, resolve_tests,
    parse_tests, prepare_interaction, compute_acat_pvalues,
)

REQUIRED_ARGS = ["--plink-prefix", "geno", "--phenotypes", "expr.bed.gz", "--covariates", "cov.txt",
                 "--interaction", "int.txt", "--output-dir", "out", "--prefix", "test"]


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


# ── --tests / --cosinor-2df ─────────────────────────────────────────────────

def test_resolve_tests_default():
    pytest.importorskip("torch")
    args = build_parser().parse_args(REQUIRED_ARGS)
    assert resolve_tests(args) == ["2df"]


def test_resolve_tests_list():
    pytest.importorskip("torch")
    args = build_parser().parse_args(REQUIRED_ARGS + ["--tests", "4df, 2df,4df"])
    assert resolve_tests(args) == ["4df", "2df"]


def test_cosinor_2df_alias_warns(capsys):
    pytest.importorskip("torch")
    args = build_parser().parse_args(REQUIRED_ARGS + ["--cosinor-2df"])
    assert resolve_tests(args) == ["2df"]
    assert "deprecated" in capsys.readouterr().err


def test_parse_tests_unknown_raises():
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="6df"):
        parse_tests("2df,6df")


# ── interaction columns and harmonics ───────────────────────────────────────

def _interaction(columns):
    return pd.DataFrame(np.zeros((3, len(columns))), index=["S1", "S2", "S3"], columns=columns)


def test_prepare_interaction_4df_missing_sin2_raises():
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="sin2_t"):
        prepare_interaction(_interaction(["cos_t", "sin_t", "cos2_t"]), ["4df"])


def test_prepare_interaction_2df_uses_harmonics_1_model():
    pytest.importorskip("torch")
    out = prepare_interaction(_interaction(["cos_t", "sin_t", "cos2_t", "sin2_t"]), ["2df", "3df"])
    assert list(out.columns) == ["cos_t", "sin_t"]


def test_prepare_interaction_4df_uses_harmonics_2_model():
    pytest.importorskip("torch")
    out = prepare_interaction(_interaction(["sin_t", "cos_t", "sin2_t", "cos2_t"]), ["2df", "4df"])
    assert list(out.columns) == ["cos_t", "sin_t", "cos2_t", "sin2_t"]


def test_check_no_overlap_raises():
    cov = pd.DataFrame(np.zeros((3, 2)), index=["S1", "S2", "S3"], columns=["PC1", "cos_t"])
    with pytest.raises(ValueError, match="cos_t"):
        check_no_overlap(cov, _interaction(["cos_t", "sin_t"]))


def test_compute_acat_pvalues(tmp_path):
    pytest.importorskip("pyarrow")
    df = pd.DataFrame({"pval_g_x_cos_t": [0.01, 0.5], "pval_g_x_sin_t": [0.2, 0.9]})
    df.to_parquet(tmp_path / "test.cis_qtl_pairs.chr1.parquet")
    compute_acat_pvalues(str(tmp_path), "test")
    out = pd.read_parquet(tmp_path / "test.cis_qtl_pairs.chr1.parquet")
    assert "pval_g_x_cosinor_acat" in out
    assert "pval_g_x_cosinor_2df" not in out
    assert ((out["pval_g_x_cosinor_acat"] > 0) & (out["pval_g_x_cosinor_acat"] < 1)).all()
