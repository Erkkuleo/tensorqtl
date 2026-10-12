"""cis.map_nominal with joint cosinor tests (synthetic data, CPU)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('pyarrow')
sys.path.insert(0, os.path.dirname(__file__))
from cosinor_sim import fit_joint, make_cis_data  # noqa: E402

import cis  # noqa: E402
import core  # noqa: E402
import cosinor  # noqa: E402
import eigenmt  # noqa: E402

N_COV = 3


def _run(tmp_path, data, **kwargs):
    genotype_df, variant_df, phenotype_df, phenotype_pos_df, covariates_df, interaction_df = data
    kwargs.setdefault('maf_threshold_interaction', 0)
    top = cis.map_nominal(genotype_df, variant_df, phenotype_df, phenotype_pos_df, 'sim',
                          covariates_df=covariates_df, interaction_df=interaction_df,
                          output_dir=str(tmp_path), write_top=False, verbose=False, **kwargs)
    pairs = pd.read_parquet(os.path.join(tmp_path, 'sim.cis_qtl_pairs.chr1.parquet'))
    return pairs, top


def test_map_nominal_joint_columns_and_top(tmp_path):
    rng = np.random.default_rng(10)
    data = make_cis_data(rng, harmonics=2)
    genotype_df, _, phenotype_df, _, covariates_df, interaction_df = data
    tests = ['2df', '4df', '5df']
    joint = cosinor.build_joint_tests(tests, interaction_df.columns)
    assert list(joint) == ['cosinor_2df_h2', 'cosinor_4df_h2', 'cosinor_5df_h2']
    pairs, top = _run(tmp_path, data, joint_tests=joint, top_by='cosinor_2df_h2')

    # per-term columns keep their names
    for c in ['pval_g', 'pval_g_x_cos_t', 'b_g_x_sin_t', 'pval_g_x_sin2_t']:
        assert c in pairs
    dof = phenotype_df.shape[1] - 2 - N_COV - 2 * interaction_df.shape[1]
    for gene in phenotype_df.index:
        sub = pairs[pairs['phenotype_id'] == gene].set_index('variant_id')
        _, _, F, dof_ = fit_joint(genotype_df.loc[sub.index].values, phenotype_df.loc[gene].values,
                                  covariates_df.values, interaction_df, tests)
        assert dof_ == dof
        for t in tests:
            label = f'cosinor_{t}_h2'
            np.testing.assert_allclose(sub[f'F_{label}'], F[t], rtol=1e-3, atol=1e-4)
            np.testing.assert_allclose(sub[f'pval_{label}'],
                                       cosinor.get_f_pval(sub[f'F_{label}'].values, len(joint[label]), dof))
        # top association = smallest joint p-value
        assert top.loc[gene, 'variant_id'] == sub['pval_cosinor_2df_h2'].idxmin()
        assert top.loc[gene, 'num_var'] == len(sub)
        for t in tests:
            label = f'cosinor_{t}_h2'
            assert top.loc[gene, f'pval_{label}'] == pytest.approx(sub.loc[top.loc[gene, 'variant_id'], f'pval_{label}'])

    # planted g:sin_t effect on gene0 at the second variant
    assert top.loc['gene0', 'variant_id'] == genotype_df.index[1]

    bonf = np.minimum(1, top['pval_cosinor_2df_h2'] * top['num_var'])
    np.testing.assert_allclose(top['pval_cosinor_2df_h2_bonf'], bonf)
    np.testing.assert_allclose(top['qval_cosinor_2df_h2'], eigenmt.padjust_bh(bonf.values))


def test_map_nominal_without_joint_tests_unchanged_columns(tmp_path):
    rng = np.random.default_rng(11)
    data = make_cis_data(rng, harmonics=1)
    pairs, top = _run(tmp_path, data)
    assert not [c for c in pairs.columns if c.startswith(('F_', 'pval_cosinor', 'qval_'))]
    assert 'num_var' not in top
    assert list(pairs.columns) == [
        'phenotype_id', 'variant_id', 'start_distance', 'af', 'ma_samples', 'ma_count',
        'pval_g', 'b_g', 'b_g_se',
        'pval_cos_t', 'b_cos_t', 'b_cos_t_se', 'pval_g_x_cos_t', 'b_g_x_cos_t', 'b_g_x_cos_t_se',
        'pval_sin_t', 'b_sin_t', 'b_sin_t_se', 'pval_g_x_sin_t', 'b_g_x_sin_t', 'b_g_x_sin_t_se',
    ]


def test_map_nominal_eigenmt_uses_top_by(tmp_path):
    rng = np.random.default_rng(12)
    data = make_cis_data(rng, harmonics=1)
    joint = cosinor.build_joint_tests(['2df'], data[5].columns)
    _, top = _run(tmp_path, data, joint_tests=joint, top_by='cosinor_2df_h1', run_eigenmt=True)
    expected = np.minimum(top['tests_emt'] * top['pval_cosinor_2df_h1'], 1)
    np.testing.assert_allclose(top['pval_emt'], expected)


# ── 6. error paths and filters ──────────────────────────────────────────────

@pytest.mark.parametrize('maf_threshold_interaction', [0, 0.05])
def test_map_nominal_monomorphic_variant_is_filtered(tmp_path, maf_threshold_interaction):
    rng = np.random.default_rng(13)
    data = make_cis_data(rng, harmonics=2, monomorphic_ix=3)
    joint = cosinor.build_joint_tests(['4df'], data[5].columns)
    pairs, top = _run(tmp_path, data, joint_tests=joint, top_by='cosinor_4df_h2',
                      maf_threshold_interaction=maf_threshold_interaction)
    assert data[0].index[3] not in set(pairs['variant_id'])
    assert np.isfinite(pairs['F_cosinor_4df_h2']).all()
    assert len(top) == data[2].shape[0]


def test_filter_maf_interaction_bins():
    # 8 samples in 4 bins of 2
    bins = torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 1]], dtype=torch.bool)
    G = torch.tensor([[1, 0, 1, 0, 1, 0, 1, 0],   # one carrier per bin -> kept
                      [1, 1, 0, 0, 0, 0, 0, 0],   # carriers only in bin 0 -> removed
                      [0, 0, 0, 0, 0, 0, 0, 0],   # monomorphic -> removed
                      [2, 1, 2, 1, 2, 1, 2, 1]],  # minor allele = ref; carriers are g < 1.5 -> kept
                     dtype=torch.float32)
    _, mask = core.filter_maf_interaction_bins(G, bins, maf_threshold_interaction=0.05, min_carriers=1)
    assert mask.tolist() == [True, False, False, True]
    _, mask = core.filter_maf_interaction_bins(G, bins, maf_threshold_interaction=0.05, min_carriers=2)
    assert mask.tolist() == [False, False, False, False]


def test_map_nominal_quadrant_filter(tmp_path):
    rng = np.random.default_rng(14)
    data = make_cis_data(rng, harmonics=1)
    genotype_df, interaction_df = data[0], data[5]
    angle = np.mod(np.arctan2(interaction_df['sin_t'], interaction_df['cos_t']), 2 * np.pi)
    q0 = np.where(angle < np.pi / 2)[0]
    g = np.zeros(genotype_df.shape[1], dtype=np.float32)
    g[q0[:20]] = 1  # MAF ~0.07 overall, but all carriers in the first quadrant
    genotype_df.iloc[7] = g
    vid = genotype_df.index[7]
    joint = cosinor.build_joint_tests(['2df'], interaction_df.columns)

    pairs, _ = _run(tmp_path, data, joint_tests=joint, maf_threshold_interaction=0.05)
    assert vid not in set(pairs['variant_id'])
    pairs, _ = _run(tmp_path, data, joint_tests=joint, maf_threshold_interaction=0)
    assert vid in set(pairs['variant_id'])


def test_map_nominal_top_by_must_be_a_joint_test(tmp_path):
    rng = np.random.default_rng(15)
    data = make_cis_data(rng, harmonics=1)
    joint = cosinor.build_joint_tests(['2df'], data[5].columns)
    with pytest.raises(ValueError, match='top_by'):
        _run(tmp_path, data, joint_tests=joint, top_by='cosinor_4df_h2')


def test_map_nominal_invalid_joint_indices(tmp_path):
    rng = np.random.default_rng(16)
    data = make_cis_data(rng, harmonics=1)
    with pytest.raises(ValueError, match='Invalid column indices'):
        _run(tmp_path, data, joint_tests={'bad': [3, 9]})


def test_map_nominal_joint_tests_require_interaction(tmp_path):
    rng = np.random.default_rng(17)
    genotype_df, variant_df, phenotype_df, phenotype_pos_df, covariates_df, _ = make_cis_data(rng)
    with pytest.raises(ValueError, match='interaction_df'):
        cis.map_nominal(genotype_df, variant_df, phenotype_df, phenotype_pos_df, 'sim',
                        covariates_df=covariates_df, joint_tests={'cosinor_2df_h1': [3, 4]},
                        output_dir=str(tmp_path), verbose=False)
