#!/usr/bin/env python3
"""Time cis.map_nominal with and without the joint cosinor tests.

The run without joint tests uses the same code path as master (float32
inverse), so it is the baseline. Times are wall-clock seconds for one call
of map_nominal (write_stats=False, so disk I/O is excluded).

Example (bundled chr18 data):
    python scripts/benchmark_joint_wald.py \\
        --plink-prefix example/data/GEUVADIS.445_samples.GRCh38.20170504.maf01.filtered.nodup.chr18 \\
        --phenotypes example/data/GEUVADIS.445_samples.expression.bed.gz \\
        --covariates example/data/GEUVADIS.445_samples.covariates.txt \\
        --chrom chr18
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tensorqtl"))
import cis  # noqa: E402
import cosinor  # noqa: E402
import genotypeio  # noqa: E402
from core import read_phenotype_bed  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plink-prefix", required=True)
    parser.add_argument("--phenotypes", required=True)
    parser.add_argument("--covariates", required=True, help="rows = covariates, columns = samples")
    parser.add_argument("--chrom", required=True, help="phenotype chromosome to map, e.g. chr18")
    parser.add_argument("--n-phenotypes", type=int, default=None, help="limit the number of phenotypes")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()

    import torch
    print(f"device: {'cuda (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'cpu'}")

    phenotype_df, phenotype_pos_df = read_phenotype_bed(args.phenotypes)
    m = (phenotype_pos_df["chr"] == args.chrom).values
    phenotype_df, phenotype_pos_df = phenotype_df[m], phenotype_pos_df[m]
    if args.n_phenotypes:
        phenotype_df = phenotype_df.iloc[:args.n_phenotypes]
        phenotype_pos_df = phenotype_pos_df.iloc[:args.n_phenotypes]
    genotype_df, variant_df = genotypeio.load_genotypes(args.plink_prefix, select_samples=phenotype_df.columns.tolist())
    covariates_df = pd.read_csv(args.covariates, sep="\t", index_col=0).T.loc[phenotype_df.columns]
    hours = np.random.default_rng(0).uniform(0, 24, phenotype_df.shape[1])
    w = 2 * np.pi * hours / 24
    interaction = {
        1: pd.DataFrame({"cos_t": np.cos(w), "sin_t": np.sin(w)}, index=phenotype_df.columns),
        2: pd.DataFrame({"cos_t": np.cos(w), "sin_t": np.sin(w), "cos2_t": np.cos(2 * w),
                         "sin2_t": np.sin(2 * w)}, index=phenotype_df.columns),
    }
    print(f"{phenotype_df.shape[0]} phenotypes, {phenotype_df.shape[1]} samples, {variant_df.shape[0]} variants")

    runs = [
        ("h1, no joint tests (master path)", 1, None),
        ("h1, 2df", 1, ["2df"]),
        ("h2, no joint tests (master path)", 2, None),
        ("h2, 2df+4df+5df", 2, ["2df", "4df", "5df"]),
    ]
    results = []
    for name, h, tests in runs:
        joint = cosinor.build_joint_tests(tests, interaction[h].columns) if tests else None
        times = []
        for _ in range(args.repeats):
            t0 = time.time()
            cis.map_nominal(genotype_df, variant_df, phenotype_df, phenotype_pos_df, "bench",
                            covariates_df=covariates_df, interaction_df=interaction[h],
                            maf_threshold_interaction=0, joint_tests=joint,
                            top_by=next(iter(joint)) if joint else None,
                            write_top=False, write_stats=False, verbose=False)
            times.append(time.time() - t0)
        results.append((name, min(times)))

    print("\nruntime (s, best of repeats):")
    for name, t in results:
        print(f"  {name:<36} {t:8.1f}")
    print(f"  h1 overhead: {results[1][1] / results[0][1] - 1:+.0%}")
    print(f"  h2 overhead: {results[3][1] / results[2][1] - 1:+.0%}")


if __name__ == "__main__":
    main()
