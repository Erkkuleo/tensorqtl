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
