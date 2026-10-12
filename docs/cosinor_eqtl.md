# Cosinor eQTL Mapping with tensorQTL

## Biological Rationale

Gene expression levels vary with time of day in many tissues, driven by circadian
transcriptional programs. When samples are collected across a range of collection
times, two distinct biological questions arise:

1. **Confounding control**: time of day affects expression globally; without
   controlling for it, circadian genes appear differentially expressed across any
   factor that correlates with collection time (batch, season, study phase).
   In this pipeline the time terms enter the model as interaction main effects
   (see below), which removes this confounding.

2. **Interaction testing**: does a SNP's effect on expression *depend on* the time
   of day? A genotype × circadian rhythm interaction indicates that a variant
   affects the amplitude, phase or shape of a gene's daily rhythm. This is the
   cosinor eQTL test.

The cosinor model encodes time t (hours, period P = 24 h) as

    c1 = cos_t  = cos(2πt/P)     s1 = sin_t  = sin(2πt/P)      (24 h rhythm)
    c2 = cos2_t = cos(4πt/P)     s2 = sin2_t = sin(4πt/P)      (12 h harmonic)

Any 24 h rhythm with amplitude A and peak time φ can be written as

    A·cos(ωt − φ) = (A cos φ)·cos(ωt) + (A sin φ)·sin(ωt)

so a genotype effect that peaks at an arbitrary time is split between the
SNP×cos_t and SNP×sin_t coefficients.

## Models

Covariates are partialled out of all terms (Frisch–Waugh–Lovell), as tensorQTL
does for every model. tensorQTL enters each interaction column as a main
effect, so **cos_t/sin_t must not also be in the covariates file** — that makes
the design matrix singular. Both `cosinor_preprocess.py` and
`run_cosinor_qtl.py` stop with an error if they are.

    harmonics = 1:  y ~ g + c1 + s1 + g:c1 + g:s1 + covariates
    harmonics = 2:  y ~ g + c1 + s1 + c2 + s2 + g:c1 + g:s1 + g:c2 + g:s2 + covariates

## Joint tests

| Test | H0 (coefficients set to zero) | q (numerator DF) | Biological meaning |
|---|---|---|---|
| `2df` | g:c1, g:s1 | 2 | 2-DF (2-DOF) joint sin/cos test: genotype changes the 24 h amplitude and/or phase, wherever the peak falls |
| `3df` | g, g:c1, g:s1 | 3 | mesor shift and/or 24 h rhythm change |
| `4df` | g:c1, g:s1, g:c2, g:s2 | 4 | genotype changes rhythm shape at 24 h or 12 h |
| `5df` | g, g:c1, g:s1, g:c2, g:s2 | 5 | any genotype effect: mesor shift and/or rhythm change |

The presets live in `COSINOR_TESTS` in `tensorqtl/cosinor.py`. The math works
on any set of design columns, so the definitions can be changed there without
touching the test code.

**Which model a test comes from.** If only `2df`/`3df` are requested, they are
computed from the harmonics = 1 model. If `4df` or `5df` is requested, all
requested tests come from the single harmonics = 2 fit; `2df` there is the 24 h
interaction adjusted for the 12 h interaction. Column names record the model:
`pval_cosinor_2df_h1` vs `pval_cosinor_2df_h2`.

### The test statistic

For each variant–gene pair, with design columns `X = [g, i_1..i_ni, g:i_1..g:i_ni]`
(after residualizing on covariates), coefficients `b`, `(XᵀX)⁻¹`, residual sum
of squares `rss` and residual degrees of freedom `dof`, the joint Wald F-test
of the index set S (size q) is

    W = b_Sᵀ [(XᵀX)⁻¹]_SS⁻¹ b_S
    F = (W / q) / (rss / dof)
    p = P(F_{q, dof} > F)

This is the classical F-test for nested OLS models and is exact under normal
errors. `dof = n − 2 − k − 2·ni` (intercept, g, k covariates, ni interaction
main effects and ni genotype interactions); harmonics = 2 costs 4 more degrees
of freedom than harmonics = 1. With a phenotype-specific (paired) covariate,
`dof` drops by one for that phenotype. The computation runs in float64.

### Why the ACAT p-value was replaced

The earlier "2-DF" p-value (`pval_g_x_cosinor_2df`) combined the two 1-DF
p-values of SNP×cos_t and SNP×sin_t with ACAT. It is a valid p-value
combination, but it is not a test of H0: β(g:cos) = β(g:sin) = 0. It ignores
the covariance of the two estimates, and its power depends on the phase of the
effect. The joint F-test uses the full covariance and is phase invariant:
shifting all sample times by a constant rotates (cos_t, sin_t) and leaves the
2-DF F statistic unchanged. A cos-only 1-DF test loses power when the peak is
near 6 h or 18 h. `run_cosinor_qtl.py --also-acat` still writes the old
combination as `pval_g_x_cosinor_acat`, for comparison only.

### Caveats

- **`5df` (and `3df`) include the SNP main effect.** Ordinary, non-rhythmic
  eQTLs make them significant without any change in rhythm. Read them
  alongside `pval_g` and `2df`/`4df`. The null-calibration test in
  `tests/test_cosinor_joint.py` documents this: with a main eQTL effect and no
  interaction, `2df`/`4df` are calibrated and `5df` is not.
- **Inferred phases carry error.** Phases estimated with CHIRAL (or any other
  method) have estimation error that these tests do not propagate. Error in t
  is error in the interaction regressors, which biases interaction effects
  toward zero (errors-in-variables attenuation). This costs power, and effect
  sizes are underestimated.
- **Gene-level significance.** `pval_*_bonf` is a Bonferroni correction over
  the cis variants of one gene; the q-values are BH across genes on those
  values. Permutation-based gene-level p-values for the joint tests (permuting
  sample times) are not implemented yet.

## Output columns

Per variant–gene pair (`<prefix>.cis_qtl_pairs.<chr>.parquet`), besides the
standard tensorQTL per-term columns:

| Column | Description |
|---|---|
| `pval_g`, `b_g`, `b_g_se` | Main SNP effect |
| `pval_cos_t`, `b_cos_t`, `b_cos_t_se` (and `sin_t`, `cos2_t`, `sin2_t`) | Main time-term effects |
| `pval_g_x_cos_t`, `b_g_x_cos_t`, `b_g_x_cos_t_se` (and the other terms) | 1-DF SNP × term interactions |
| `F_cosinor_<test>_h<k>` | Joint F statistic |
| `pval_cosinor_<test>_h<k>` | Joint p-value (−log10 if `logp=True`) |

Top association per gene (`<prefix>.cis_qtl_top_assoc.txt.gz`), chosen by the
smallest p-value of the **first** test in `--tests`:

| Column | Description |
|---|---|
| `num_var` | Number of cis variants tested for the gene |
| `F_cosinor_<test>_h<k>`, `pval_cosinor_<test>_h<k>` | Joint test of the top variant |
| `pval_cosinor_<test>_h<k>_bonf` | `min(1, p × num_var)` |
| `qval_cosinor_<test>_h<k>` | BH across genes on the Bonferroni-adjusted values |

With `run_eigenmt=True`, `pval_emt` uses the p-value of the test that selected
the top variant.

## Variant filtering

With cos_t/sin_t interactions and `maf_threshold_interaction > 0` (default
0.05), a variant is tested only if it has MAF ≥ the threshold overall and at
least `min_bin_carriers` (default 5) minor-allele carriers in each time-of-day
quadrant, with quadrants defined by `atan2(sin_t, cos_t)`. The log reports how
many variant–gene pairs the filter removed. Monomorphic variants are always
removed. Set `maf_threshold_interaction=0` to keep only the monomorphic filter
(this was the behaviour for more than one interaction term before).

## Workflow

### Step 1: Prepare cosinor terms

```bash
python scripts/cosinor_preprocess.py \
  --metadata sample_metadata.tsv \
  --covariates existing_covariates.txt \
  --out-covariates covariates.txt \
  --out-interaction interaction.txt \
  --time-col hour \
  --harmonics 2          # 1 = cos_t, sin_t; 2 = also cos2_t, sin2_t (for 4df/5df)
```

`sample_metadata.tsv` must be tab-separated with sample IDs in the first column
and a time column (default `hour`: numeric hours 0–24 or ISO 8601 timestamps).
`--mode toy` encodes time of year instead (period 365.25 days).

### Step 2: Run cosinor eQTL mapping

```bash
python scripts/run_cosinor_qtl.py \
  --plink-prefix path/to/genotypes \
  --phenotypes expression.bed.gz \
  --covariates covariates.txt \
  --interaction interaction.txt \
  --output-dir results/ \
  --prefix my_study.cosinor \
  --tests 2df,4df
```

`--cosinor-2df` is a deprecated alias for `--tests 2df`.

### Python API

```python
import cis, cosinor
joint_tests = cosinor.build_joint_tests(['2df', '4df'], interaction_df.columns)
# {'cosinor_2df_h2': [5, 6], 'cosinor_4df_h2': [5, 6, 7, 8]}
cis.map_nominal(genotype_df, variant_df, phenotype_df, phenotype_pos_df, prefix,
                covariates_df=covariates_df, interaction_df=interaction_df,
                joint_tests=joint_tests, top_by='cosinor_2df_h2')
```

`joint_tests` takes any `{name: [column indices]}` into `[g, i_1..i_ni, g:i_1..g:i_ni]`,
so other joint hypotheses can be tested the same way.

## Follow-ups (not implemented)

- Permutation-based gene-level p-values for the joint tests (permute sample
  times, not genotypes).
- Conditionally independent mapping for interaction QTLs.
- Joint tests in trans mode and for grouped phenotypes (`group_s`).
