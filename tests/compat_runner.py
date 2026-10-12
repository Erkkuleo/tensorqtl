"""Run cis.map_nominal from a given tensorqtl checkout on the bundled chr18 example.

Used by test_backward_compat.py to compare this branch against master:
    python tests/compat_runner.py --repo <checkout> --data-dir example/data --out-dir <dir> --ni 2
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--repo', required=True)
parser.add_argument('--data-dir', required=True)
parser.add_argument('--out-dir', required=True)
parser.add_argument('--ni', type=int, choices=[1, 2], default=1)
parser.add_argument('--maf-threshold-interaction', type=float, default=0.05)
parser.add_argument('--n-phenotypes', type=int, default=20)
args = parser.parse_args()

sys.path.insert(0, os.path.join(os.path.abspath(args.repo), 'tensorqtl'))
import cis  # noqa: E402
import genotypeio  # noqa: E402
from core import read_phenotype_bed  # noqa: E402

phenotype_df, phenotype_pos_df = read_phenotype_bed(
    os.path.join(args.data_dir, 'GEUVADIS.445_samples.expression.bed.gz'))
m = (phenotype_pos_df['chr'] == 'chr18').values
phenotype_df = phenotype_df[m].iloc[:args.n_phenotypes]
phenotype_pos_df = phenotype_pos_df[m].iloc[:args.n_phenotypes]

covariates_df = pd.read_csv(os.path.join(args.data_dir, 'GEUVADIS.445_samples.covariates.txt'),
                            sep='\t', index_col=0).T.loc[phenotype_df.columns]
genotype_df, variant_df = genotypeio.load_genotypes(
    os.path.join(args.data_dir, 'GEUVADIS.445_samples.GRCh38.20170504.maf01.filtered.nodup.chr18'))

hours = np.random.default_rng(0).uniform(0, 24, phenotype_df.shape[1])
interaction_df = pd.DataFrame({'cos_t': np.cos(2 * np.pi * hours / 24),
                               'sin_t': np.sin(2 * np.pi * hours / 24)},
                              index=phenotype_df.columns).iloc[:, :args.ni]

os.makedirs(args.out_dir, exist_ok=True)
cis.map_nominal(genotype_df, variant_df, phenotype_df, phenotype_pos_df, 'compat',
                covariates_df=covariates_df, interaction_df=interaction_df,
                maf_threshold_interaction=args.maf_threshold_interaction,
                output_dir=args.out_dir, verbose=False)
