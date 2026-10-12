"""Joint Wald F-tests for cosinor (circadian) genotype x time interactions.

The interaction model fitted by calculate_interaction_nominal has design
columns [g, i_1..i_ni, g:i_1..g:i_ni] after covariates are partialled out.
A joint test sets an arbitrary subset S of these coefficients to zero:

    W = b_S' (Xinv_SS)^-1 b_S,    F = (W / q) / (rss / dof),    q = |S|

which is the classical F-test for nested OLS models.

Test presets live in COSINOR_TESTS so the definitions can be changed without
touching the math. Terms are named 'g' (genotype main effect), '<col>'
(interaction main effect) and 'g_x_<col>' (genotype x interaction column).
"""
import numpy as np
import scipy.stats as stats
import torch


# name -> terms set to zero under H0, and the number of harmonics required
COSINOR_TESTS = {
    # 2-DF (2-DOF) joint sin/cos test: genotype changes 24 h amplitude and/or phase
    '2df': {'terms': ['g_x_cos_t', 'g_x_sin_t'], 'harmonics': 1},
    # mesor shift and/or 24 h rhythm change
    '3df': {'terms': ['g', 'g_x_cos_t', 'g_x_sin_t'], 'harmonics': 1},
    # rhythm shape change at 24 h and/or 12 h
    '4df': {'terms': ['g_x_cos_t', 'g_x_sin_t', 'g_x_cos2_t', 'g_x_sin2_t'], 'harmonics': 2},
    # any genotype effect: mesor shift and/or rhythm change
    '5df': {'terms': ['g', 'g_x_cos_t', 'g_x_sin_t', 'g_x_cos2_t', 'g_x_sin2_t'], 'harmonics': 2},
}

HARMONIC_COLUMNS = {
    1: ['cos_t', 'sin_t'],
    2: ['cos_t', 'sin_t', 'cos2_t', 'sin2_t'],
}


def model_harmonics(test_names):
    """Number of harmonics the model must include to compute all requested tests."""
    unknown = [t for t in test_names if t not in COSINOR_TESTS]
    if unknown:
        raise ValueError(f"Unknown cosinor test(s) {unknown}. Available: {list(COSINOR_TESTS)}")
    return max(COSINOR_TESTS[t]['harmonics'] for t in test_names)


def joint_test_label(test_name, harmonics):
    """Output label of a test, e.g. 'cosinor_2df_h1'; columns are F_<label>, pval_<label>."""
    return f'cosinor_{test_name}_h{harmonics}'


def term_index(term, interaction_columns):
    """Column index of a term in X = [g, i_1..i_ni, g:i_1..g:i_ni]."""
    interaction_columns = list(interaction_columns)
    ni = len(interaction_columns)
    if term == 'g':
        return 0
    if term.startswith('g_x_'):
        v = term[len('g_x_'):]
        if v in interaction_columns:
            return 1 + ni + interaction_columns.index(v)
    elif term in interaction_columns:
        return 1 + interaction_columns.index(term)
    raise ValueError(f"Term '{term}' requires interaction column '{term.replace('g_x_', '', 1)}', "
                     f"which is not among the interaction columns {interaction_columns}.")


def resolve_test_indices(test_names, interaction_columns, tests=None):
    """Map test names to X column indices.

    Args:
        test_names: e.g. ['2df', '4df']
        interaction_columns: interaction_df.columns, in order
        tests: preset dictionary (default: COSINOR_TESTS)

    Returns:
        dict {test_name: [column indices]}
    """
    if tests is None:
        tests = COSINOR_TESTS
    out = {}
    for name in test_names:
        if name not in tests:
            raise ValueError(f"Unknown test '{name}'. Available: {list(tests)}")
        out[name] = [term_index(t, interaction_columns) for t in tests[name]['terms']]
    return out


def build_joint_tests(test_names, interaction_columns):
    """Joint tests keyed by output label, for cis.map_nominal(joint_tests=...).

    All tests are computed from a single fit; the label records the number
    of harmonics in that fit (e.g. 'cosinor_2df_h2' is the 24 h interaction
    adjusted for the 12 h interaction).
    """
    columns = list(interaction_columns)
    harmonics = 2 if all(c in columns for c in HARMONIC_COLUMNS[2]) else 1
    idx = resolve_test_indices(test_names, columns)
    return {joint_test_label(name, harmonics): idx[name] for name in test_names}


def joint_wald_f(b_t, Xinv_t, rss_t, dof, idx):
    """Joint Wald F statistic for H0: b[idx] = 0, vectorized over variants.

    Args:
        b_t:    ng x nb coefficients
        Xinv_t: ng x nb x nb, (X'X)^-1
        rss_t:  ng residual sums of squares
        dof:    residual degrees of freedom
        idx:    list of column indices (size q)

    Returns:
        F statistics (ng, float64)
    """
    idx_t = torch.as_tensor(idx, dtype=torch.long, device=b_t.device)
    q = len(idx)
    b_s = b_t.double()[:, idx_t]  # ng x q
    V_s = Xinv_t.double()[:, idx_t][:, :, idx_t]  # ng x q x q
    w_t = (b_s.unsqueeze(1) @ torch.linalg.solve(V_s, b_s.unsqueeze(-1))).reshape(-1)  # ng
    return (w_t / q) / (rss_t.double() / dof)


def get_f_pval(F, q, dof, log=False):
    """p-value for an F statistic with (q, dof) degrees of freedom; -log10(p) if log=True.

    dof can be an array (e.g., phenotype-specific dof with paired covariates).
    """
    if not log:
        return stats.f.sf(F, q, dof)
    else:
        return -stats.f.logsf(F, q, dof) / np.log(10)
