"""Joint Wald F-tests for cosinor interactions (tensorqtl/cosinor.py, core, cis)."""
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip('torch')
sys.path.insert(0, os.path.dirname(__file__))
from cosinor_sim import cosinor_terms, simulate_genotypes, fit_joint  # noqa: E402

import core  # noqa: E402
import cosinor  # noqa: E402


# ── index resolution ────────────────────────────────────────────────────────

def test_resolve_test_indices_harmonics_1():
    idx = cosinor.resolve_test_indices(['2df', '3df'], ['cos_t', 'sin_t'])
    # X = [g, cos_t, sin_t, g:cos_t, g:sin_t]
    assert idx == {'2df': [3, 4], '3df': [0, 3, 4]}


def test_resolve_test_indices_harmonics_2():
    idx = cosinor.resolve_test_indices(['2df', '4df', '5df'], ['cos_t', 'sin_t', 'cos2_t', 'sin2_t'])
    # X = [g, c1, s1, c2, s2, g:c1, g:s1, g:c2, g:s2]
    assert idx == {'2df': [5, 6], '4df': [5, 6, 7, 8], '5df': [0, 5, 6, 7, 8]}


def test_resolve_test_indices_missing_column():
    with pytest.raises(ValueError, match="sin2_t"):
        cosinor.resolve_test_indices(['4df'], ['cos_t', 'sin_t', 'cos2_t'])


def test_resolve_test_indices_unknown_test():
    with pytest.raises(ValueError, match="Unknown"):
        cosinor.resolve_test_indices(['7df'], ['cos_t', 'sin_t'])


def test_build_joint_tests_labels():
    assert cosinor.build_joint_tests(['2df'], ['cos_t', 'sin_t']) == {'cosinor_2df_h1': [3, 4]}
    assert cosinor.build_joint_tests(['2df', '4df'], ['cos_t', 'sin_t', 'cos2_t', 'sin2_t']) == \
        {'cosinor_2df_h2': [5, 6], 'cosinor_4df_h2': [5, 6, 7, 8]}


def test_model_harmonics():
    assert cosinor.model_harmonics(['2df', '3df']) == 1
    assert cosinor.model_harmonics(['2df', '4df']) == 2


# ── 1. exactness vs statsmodels ─────────────────────────────────────────────

@pytest.mark.parametrize('harmonics', [1, 2])
def test_joint_f_matches_statsmodels(harmonics):
    sm = pytest.importorskip('statsmodels.api')
    rng = np.random.default_rng(1)
    n, k, m = 200, 3, 20
    I = cosinor_terms(rng.uniform(0, 24, n), harmonics)
    C = rng.normal(size=(n, k))
    G = simulate_genotypes(rng, m, n)
    y = (C @ rng.normal(size=k) + 0.5 * I['cos_t'].values + 0.3 * G[0]
         + 0.4 * G[0] * I['sin_t'].values + rng.normal(size=n))
    tests = [t for t, d in cosinor.COSINOR_TESTS.items() if d['harmonics'] <= harmonics]
    _, _, F, dof = fit_joint(G, y, C, I, tests)
    indices = cosinor.resolve_test_indices(tests, I.columns)

    for j in range(m):
        g = G[j]
        # full model with explicit covariates: [1, C, g, I, g*I]
        X = np.column_stack([np.ones(n), C, g, I.values, g[:, None] * I.values])
        fit = sm.OLS(y, X).fit()
        assert fit.df_resid == dof
        for name in tests:
            idx = indices[name]
            R = np.zeros((len(idx), X.shape[1]))
            for r, c in enumerate(idx):
                R[r, 1 + k + c] = 1
            ft = fit.f_test(R)
            assert F[name][j] == pytest.approx(float(np.squeeze(ft.fvalue)), rel=1e-5)
            p = cosinor.get_f_pval(F[name][j], len(idx), dof)
            assert p == pytest.approx(float(np.squeeze(ft.pvalue)), rel=1e-5, abs=1e-300)


def test_joint_f_float32_inputs_close_to_float64():
    """map_nominal passes float32 tensors; the joint test is still computed in float64."""
    rng = np.random.default_rng(2)
    n = 200
    I = cosinor_terms(rng.uniform(0, 24, n), 2)
    C = rng.normal(size=(n, 3))
    G = simulate_genotypes(rng, 20, n)
    y = rng.normal(size=n) + 0.5 * G[0] * I['cos2_t'].values
    _, _, F64, _ = fit_joint(G, y, C, I, ['2df', '4df', '5df'], dtype=torch.float64)
    _, _, F32, _ = fit_joint(G, y, C, I, ['2df', '4df', '5df'], dtype=torch.float32)
    for name in F64:
        assert F32[name].dtype == np.float64
        np.testing.assert_allclose(F32[name], F64[name], rtol=1e-3, atol=1e-3)


def test_get_f_pval_log():
    F = np.array([0.5, 3.0, 40.0])
    p = cosinor.get_f_pval(F, 2, 150)
    logp = cosinor.get_f_pval(F, 2, 150, log=True)
    np.testing.assert_allclose(logp, -np.log10(p), rtol=1e-10)


# ── nps > 1 branch ──────────────────────────────────────────────────────────

def test_multi_phenotype_b_se_matches_single_phenotype():
    """The nps > 1 branch used repeat([1,3,1]), which broke for > 1 interaction."""
    rng = np.random.default_rng(3)
    n = 120
    I = cosinor_terms(rng.uniform(0, 24, n), 1)
    G = torch.tensor(simulate_genotypes(rng, 10, n))
    Y = torch.tensor(rng.normal(size=(2, n)))
    I_t = torch.tensor(I.values)
    residualizer = core.Residualizer(torch.tensor(rng.normal(size=(n, 2))))
    _, b_m, b_se_m = core.calculate_interaction_nominal(G, Y, I_t, residualizer=residualizer)[:3]
    assert b_se_m.shape == (10, 5, 2)
    for j in range(2):
        _, b_s, b_se_s = core.calculate_interaction_nominal(G, Y[[j]], I_t, residualizer=residualizer)[:3]
        np.testing.assert_allclose(b_m[:, :, j].numpy(), b_s.numpy(), rtol=1e-6)
        np.testing.assert_allclose(b_se_m[:, :, j].numpy(), b_se_s.numpy(), rtol=1e-6)


def test_joint_tests_require_single_phenotype():
    rng = np.random.default_rng(4)
    n = 50
    I = torch.tensor(cosinor_terms(rng.uniform(0, 24, n), 1).values)
    G = torch.tensor(simulate_genotypes(rng, 3, n))
    Y = torch.tensor(rng.normal(size=(2, n)))
    with pytest.raises(NotImplementedError):
        core.calculate_interaction_nominal(G, Y, I, joint_tests={'2df': [3, 4]})


# ── 2. null calibration ─────────────────────────────────────────────────────

def _null_pvalues(seed, n_variants=5000, n=200):
    """No genotype x time effect, but a real 24 h rhythm and a real main eQTL effect."""
    rng = np.random.default_rng(seed)
    hours = rng.uniform(0, 24, n)
    I1 = cosinor_terms(hours, 1)
    I2 = cosinor_terms(hours, 2)
    C = rng.normal(size=(n, 3))
    G = simulate_genotypes(rng, n_variants, n)
    rhythm = 1.0 * I1['cos_t'].values + 0.5 * I1['sin_t'].values
    p = {'2df_h1': [], '4df_h2': [], '5df_h2': []}
    for j in range(n_variants):
        y = rhythm + 0.5 * G[j] + C @ np.array([0.3, -0.2, 0.1]) + rng.normal(size=n)
        _, _, F1, dof1 = fit_joint(G[[j]], y, C, I1, ['2df'])
        _, _, F2, dof2 = fit_joint(G[[j]], y, C, I2, ['4df', '5df'])
        p['2df_h1'].append(cosinor.get_f_pval(F1['2df'][0], 2, dof1))
        p['4df_h2'].append(cosinor.get_f_pval(F2['4df'][0], 4, dof2))
        p['5df_h2'].append(cosinor.get_f_pval(F2['5df'][0], 5, dof2))
    return {k: np.array(v) for k, v in p.items()}


def test_null_calibration():
    # 5,000 variants (spec: 2,000) so the 0.04-0.06 window is > 3 binomial SDs wide
    p = _null_pvalues(seed=5)
    for name in ['2df_h1', '4df_h2']:
        frac = np.mean(p[name] < 0.05)
        assert 0.04 <= frac <= 0.06, f'{name}: {frac:.4f} of null p-values < 0.05'
    # 5df includes the genotype main effect: ordinary eQTLs make it significant
    # even without any rhythm change, so it is NOT calibrated for interactions.
    assert np.mean(p['5df_h2'] < 0.05) > 0.5


# ── 3. phase invariance ─────────────────────────────────────────────────────

@pytest.mark.parametrize('harmonics', [1, 2])
def test_2df_is_phase_invariant(harmonics):
    rng = np.random.default_rng(6)
    n = 200
    hours = rng.uniform(0, 24, n)
    C = rng.normal(size=(n, 2))
    G = simulate_genotypes(rng, 10, n)
    I = cosinor_terms(hours, harmonics)
    y = 0.6 * G[0] * I['sin_t'].values + rng.normal(size=n)  # pure g:sin effect

    tstat0, _, F0, _ = fit_joint(G, y, C, I, ['2df'])
    for shift in [3.0, 6.0, 7.5, 13.0]:
        I_shift = cosinor_terms(hours + shift, harmonics)
        tstat, _, F, _ = fit_joint(G, y, C, I_shift, ['2df'])
        np.testing.assert_allclose(F['2df'], F0['2df'], rtol=1e-8)
        # the 1-DF g_x_cos_t statistic depends on where the peak falls
        ix_gcos = 1 + I.shape[1]
        assert not np.isclose(tstat[0, ix_gcos], tstat0[0, ix_gcos], rtol=1e-2)


# ── 4. power sanity ─────────────────────────────────────────────────────────

def test_4df_beats_2df_for_12h_interaction():
    rng = np.random.default_rng(7)
    n, reps = 200, 50
    logp_2df, logp_4df = [], []
    for _ in range(reps):
        hours = rng.uniform(0, 24, n)
        I1 = cosinor_terms(hours, 1)
        I2 = cosinor_terms(hours, 2)
        g = simulate_genotypes(rng, 1, n)
        y = g[0] * (0.4 * I2['cos2_t'].values + 0.4 * I2['sin2_t'].values) + rng.normal(size=n)
        _, _, F1, dof1 = fit_joint(g, y, None, I1, ['2df'])
        _, _, F2, dof2 = fit_joint(g, y, None, I2, ['4df'])
        logp_2df.append(cosinor.get_f_pval(F1['2df'][0], 2, dof1, log=True))
        logp_4df.append(cosinor.get_f_pval(F2['4df'][0], 4, dof2, log=True))
    assert np.median(logp_4df) > np.median(logp_2df)
