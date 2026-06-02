import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from run_cosinor_qtl import validate_sample_alignment


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
