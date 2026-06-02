#!/usr/bin/env python3
"""Cosinor preprocessing for circadian eQTL mapping."""
import argparse
import math
import numpy as np
import pandas as pd
import sys


def parse_time_to_hours(value: str) -> float:
    """Convert a time value string to fractional hours (0-24).

    Accepts either a numeric hour (e.g. "14.5") or an ISO 8601 timestamp
    (e.g. "2024-01-15T14:30:00"). ISO is tried first; if it fails, the
    value is parsed as a plain float.
    """
    from datetime import datetime
    try:
        dt = datetime.fromisoformat(value)
        return dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    except ValueError:
        pass
    try:
        return float(value)
    except (ValueError, TypeError):
        raise ValueError(
            f"Cannot convert '{value}' to hours. "
            "Expected a numeric value (e.g. '14.5') or ISO 8601 timestamp "
            "(e.g. '2024-01-15T14:30:00')."
        )


def compute_cosinor(hours: pd.Series, period: float = 24.0) -> tuple:
    """Compute cosine and sine time encodings for circadian eQTL mapping.

    Args:
        hours: Series indexed by sample ID, values are fractional hours (0-24).
        period: Cycle period in hours (default 24.0 for circadian rhythm).

    Returns:
        Tuple of (cos_t, sin_t), each a named Series indexed by sample ID.
    """
    angle = 2.0 * math.pi * hours / period
    cos_t = pd.Series(np.cos(angle.values), index=hours.index, name="cos_t")
    sin_t = pd.Series(np.sin(angle.values), index=hours.index, name="sin_t")
    return cos_t, sin_t


def load_metadata(path: str, time_col: str) -> pd.Series:
    """Load sample metadata and return hour-of-day values.

    File must be tab-separated with a header; the first column is used as
    sample IDs.

    Args:
        path: Path to TSV metadata file.
        time_col: Column name containing time-of-day values.

    Returns:
        Series of float hours indexed by sample ID.

    Raises:
        ValueError: If time_col is absent, sample IDs are duplicated,
                    or any time value cannot be converted to float.
    """
    df = pd.read_csv(path, sep="\t", index_col=0)
    if time_col not in df.columns:
        raise ValueError(
            f"time_col '{time_col}' not found in metadata. "
            f"Available columns: {list(df.columns)}"
        )
    if df.index.duplicated().any():
        dupes = df.index[df.index.duplicated()].tolist()
        raise ValueError(f"Duplicate sample IDs in metadata: {dupes}")
    return df[time_col].map(lambda v: parse_time_to_hours(str(v)))


def load_covariates(path: str) -> pd.DataFrame:
    """Load a tensorQTL-format covariates file.

    Format: rows = covariate names, columns = sample IDs (tab-separated).

    Returns:
        DataFrame with covariate names as index, sample IDs as columns.
    """
    # Read header manually to avoid a pandas 2.x crash (TypeError in
    # maybe_convert_objects) that affects files with many columns whose
    # names trigger type inference (e.g. "NA06984"-style sample IDs).
    with open(path) as f:
        sample_ids = f.readline().rstrip("\n").split("\t")[1:]
    df = pd.read_csv(path, sep="\t", skiprows=1, header=None, index_col=0)
    df.columns = sample_ids
    return df


def append_cosinor_to_covariates(
    covariates_df: pd.DataFrame, cos_t: pd.Series, sin_t: pd.Series
) -> pd.DataFrame:
    """Append cos_t and sin_t rows to a tensorQTL covariates DataFrame.

    Args:
        covariates_df: Existing covariates (rows=covariate names, cols=samples).
        cos_t: Cosine values indexed by sample ID (must cover all samples).
        sin_t: Sine values indexed by sample ID (must cover all samples).

    Returns:
        New DataFrame with cos_t and sin_t appended as the last two rows.

    Raises:
        ValueError: If 'cos_t' or 'sin_t' are already present in the index.
    """
    for name in ("cos_t", "sin_t"):
        if name in covariates_df.index:
            raise ValueError(
                f"'{name}' already present in covariates index. "
                "Remove it before re-running cosinor_preprocess.py."
            )
    new_rows = pd.DataFrame(
        [
            cos_t.reindex(covariates_df.columns).values,
            sin_t.reindex(covariates_df.columns).values,
        ],
        index=["cos_t", "sin_t"],
        columns=covariates_df.columns,
    )
    return pd.concat([covariates_df, new_rows])


def make_interaction_df(cos_t: pd.Series, sin_t: pd.Series) -> pd.DataFrame:
    """Create a tensorQTL interaction DataFrame with cos_t and sin_t terms.

    Both columns are passed to tensorQTL as interaction terms. Use
    --cosinor-2df in run_cosinor_qtl.py to compute the joint 2-DF p-value.

    Returns:
        DataFrame with sample IDs as index, columns ['cos_t', 'sin_t'].
    """
    return pd.DataFrame({"cos_t": cos_t, "sin_t": sin_t})


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute cosinor (circadian) covariates for tensorQTL interaction mapping. "
            "Appends cos_t and sin_t to an existing covariates file and writes a separate "
            "interaction file containing only cos_t."
        )
    )
    parser.add_argument("--metadata", required=True,
                        help="Sample metadata TSV. First column = sample IDs; must contain --time-col.")
    parser.add_argument("--covariates", required=True,
                        help="Existing tensorQTL covariates file (rows=covariates, cols=samples).")
    parser.add_argument("--out-covariates", required=True,
                        help="Output path for updated covariates (existing rows + cos_t + sin_t).")
    parser.add_argument("--out-interaction", required=True,
                        help="Output path for interaction file (samples x 1 column: cos_t).")
    parser.add_argument("--time-col", default="hour",
                        help="Column name for time-of-day in metadata (default: 'hour').")
    parser.add_argument("--period", type=float, default=24.0,
                        help="Cycle period in hours (default: 24.0 for circadian).")
    args = parser.parse_args()

    print(f"Loading metadata from {args.metadata}")
    hours = load_metadata(args.metadata, time_col=args.time_col)

    print(f"Loading covariates from {args.covariates}")
    covariates_df = load_covariates(args.covariates)

    covariate_samples = pd.Index(covariates_df.columns)
    missing = covariate_samples.difference(hours.index)
    if len(missing) > 0:
        shown = missing.tolist()[:10]
        suffix = f" (showing first 10 of {len(missing)})" if len(missing) > 10 else ""
        print(
            f"ERROR: {len(missing)} samples in covariates have no metadata entry{suffix}: {shown}",
            file=sys.stderr,
        )
        sys.exit(1)

    hours_aligned = hours.loc[covariate_samples]
    cos_t, sin_t = compute_cosinor(hours_aligned, period=args.period)

    updated_cov = append_cosinor_to_covariates(covariates_df, cos_t, sin_t)
    updated_cov.to_csv(args.out_covariates, sep="\t")
    print(f"Wrote updated covariates ({updated_cov.shape[0]} rows) to {args.out_covariates}")

    interaction_df = make_interaction_df(cos_t, sin_t)
    interaction_df.to_csv(args.out_interaction, sep="\t")
    print(f"Wrote interaction file ({interaction_df.shape[0]} samples) to {args.out_interaction}")


if __name__ == "__main__":
    main()
