"""Simulation helpers for the cosinor joint-test tests."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tensorqtl'))


def cosinor_terms(hours, harmonics=1, period=24.0, index=None):
    """cos_t, sin_t (and cos2_t, sin2_t) for the given times."""
    w = 2 * np.pi * np.asarray(hours, dtype=np.float64) / period
    cols = {'cos_t': np.cos(w), 'sin_t': np.sin(w)}
    if harmonics == 2:
        cols.update({'cos2_t': np.cos(2 * w), 'sin2_t': np.sin(2 * w)})
    return pd.DataFrame(cols, index=index)


def simulate_genotypes(rng, n_variants, n_samples, maf_range=(0.15, 0.5)):
    maf = rng.uniform(*maf_range, size=(n_variants, 1))
    return rng.binomial(2, maf, size=(n_variants, n_samples)).astype(np.float64)


def fit_joint(G, y, C, I, test_names, dtype=None):
    """Run core.calculate_interaction_nominal on a single phenotype.

    G: variants x samples, y: samples, C: samples x covariates (or None),
    I: interaction DataFrame (samples x interactions).

    Returns: tstat, b, F (dict test name -> F per variant), dof
    """
    import torch
    import core
    import cosinor
    if dtype is None:
        dtype = torch.float64
    t = lambda a: torch.tensor(np.asarray(a), dtype=dtype)
    residualizer = core.Residualizer(t(C)) if C is not None else None
    joint_tests = cosinor.resolve_test_indices(test_names, I.columns)
    res = core.calculate_interaction_nominal(t(G), t(y).reshape(1, -1), t(I.values),
                                             residualizer=residualizer, joint_tests=joint_tests)
    tstat, b = res[0].cpu().numpy(), res[1].cpu().numpy()
    F = {k: v.cpu().numpy() for k, v in res[6].items()}
    if residualizer is not None:
        dof = residualizer.dof - 2 * I.shape[1]
    else:
        dof = len(y) - 2 - 2 * I.shape[1]
    return tstat, b, F, dof


def make_cis_data(rng, n_samples=150, n_variants=40, n_phenotypes=3, harmonics=1, monomorphic_ix=None):
    """Small single-chromosome dataset in the format expected by cis.map_nominal."""
    samples = [f'S{i}' for i in range(n_samples)]
    pos = 10_000 + 1_000 * np.arange(n_variants)
    variant_ids = [f'chr1_{p}_A_G_b38' for p in pos]
    G = simulate_genotypes(rng, n_variants, n_samples)
    if monomorphic_ix is not None:
        G[monomorphic_ix] = 0
    genotype_df = pd.DataFrame(G.astype(np.float32), index=variant_ids, columns=samples)
    variant_df = pd.DataFrame({'chrom': 'chr1', 'pos': pos}, index=variant_ids)

    hours = rng.uniform(0, 24, n_samples)
    interaction_df = cosinor_terms(hours, harmonics, index=samples)
    covariates_df = pd.DataFrame(rng.normal(size=(n_samples, 3)), index=samples,
                                 columns=['PC1', 'PC2', 'PC3'])

    phenotype_ids = [f'gene{i}' for i in range(n_phenotypes)]
    Y = rng.normal(size=(n_phenotypes, n_samples))
    Y += rng.normal(size=(n_phenotypes, 3)) @ covariates_df.values.T
    Y += 0.5 * interaction_df['cos_t'].values
    # planted genotype x time interaction for the first phenotype
    Y[0] += 0.8 * G[1] * interaction_df['sin_t'].values + 0.3 * G[1]
    phenotype_df = pd.DataFrame(Y, index=phenotype_ids, columns=samples)
    phenotype_pos_df = pd.DataFrame({'chr': 'chr1', 'pos': 20_000 + 5_000 * np.arange(n_phenotypes)},
                                    index=phenotype_ids)
    return genotype_df, variant_df, phenotype_df, phenotype_pos_df, covariates_df, interaction_df
