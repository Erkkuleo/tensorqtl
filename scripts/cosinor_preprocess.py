#!/usr/bin/env python3
"""Cosinor preprocessing for circadian eQTL mapping."""
import argparse
import math
import numpy as np
import pandas as pd
import sys


def parse_time_to_hours(value: str) -> float:
    """Convert a time value to fractional hours (0-24) for tod mode.

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


def parse_time_to_day_of_year(value: str) -> float:
    """Convert a time value to fractional day-of-year (1-366) for toy mode.

    Accepts either a numeric day (e.g. "180.5") or an ISO 8601 timestamp
    (e.g. "2024-07-01T10:00:00"). ISO is tried first; if it fails, the
    value is parsed as a plain float.
    """
    from datetime import datetime
    try:
        dt = datetime.fromisoformat(value)
        return float(dt.timetuple().tm_yday)
    except ValueError:
        pass
    try:
        return float(value)
    except (ValueError, TypeError):
        raise ValueError(
            f"Cannot convert '{value}' to day of year. "
            "Expected a numeric value (e.g. '180.5') or ISO 8601 timestamp "
            "(e.g. '2024-07-01T10:00:00')."
        )


def compute_cosinor(hours: pd.Series, period: float = 24.0, harmonics: int = 1) -> tuple:
    """Compute cosine and sine time encodings for circadian eQTL mapping.

    Args:
        hours: Series indexed by sample ID, values are fractional hours (0-24).
        period: Cycle period in hours (default 24.0 for circadian rhythm).
        harmonics: 1 = fundamental only; 2 = also the second harmonic (period/2).

    Returns:
        (cos_t, sin_t) for harmonics=1, or (cos_t, sin_t, cos2_t, sin2_t) for
        harmonics=2; each a named Series indexed by sample ID.
    """
    if harmonics not in (1, 2):
        raise ValueError(f"harmonics must be 1 or 2, got {harmonics}")
    angle = 2.0 * math.pi * hours / period
    terms = [
        pd.Series(np.cos(angle.values), index=hours.index, name="cos_t"),
        pd.Series(np.sin(angle.values), index=hours.index, name="sin_t"),
    ]
    if harmonics == 2:
        terms += [
            pd.Series(np.cos(2.0 * angle.values), index=hours.index, name="cos2_t"),
            pd.Series(np.sin(2.0 * angle.values), index=hours.index, name="sin2_t"),
        ]
    return tuple(terms)


def load_metadata(path: str, time_col: str, mode: str = "tod") -> pd.Series:
    """Load sample metadata and return time values for cosinor encoding.

    File must be tab-separated with a header; the first column is used as
    sample IDs.

    Args:
        path: Path to TSV metadata file.
        time_col: Column name containing time values.
        mode: 'tod' (time of day, values in hours 0-24) or
              'toy' (time of year, values in day-of-year 1-366).

    Returns:
        Series of float time values indexed by sample ID.

    Raises:
        ValueError: If time_col is absent, sample IDs are duplicated,
                    or any time value cannot be parsed.
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
    parse_fn = parse_time_to_hours if mode == "tod" else parse_time_to_day_of_year
    return df[time_col].map(lambda v: parse_fn(str(v)))


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



def make_interaction_df(*terms: pd.Series) -> pd.DataFrame:
    """Create a tensorQTL interaction DataFrame from cosinor terms.

    Pass the Series returned by compute_cosinor: (cos_t, sin_t) or
    (cos_t, sin_t, cos2_t, sin2_t). All columns are passed to tensorQTL as
    interaction terms; run_cosinor_qtl.py --tests computes the joint tests.

    Returns:
        DataFrame with sample IDs as index, columns ['cos_t', 'sin_t'] or
        ['cos_t', 'sin_t', 'cos2_t', 'sin2_t'] (in the order given).
    """
    if len(terms) not in (2, 4):
        raise ValueError(f"Expected 2 or 4 cosinor terms, got {len(terms)}")
    names = ["cos_t", "sin_t", "cos2_t", "sin2_t"][:len(terms)]
    return pd.DataFrame(dict(zip(names, terms)))


def check_no_overlap(covariates_df: pd.DataFrame, interaction_df: pd.DataFrame) -> None:
    """Raise if an interaction column is also a covariate.

    tensorQTL enters every interaction column as a main effect, so the same
    term in the covariates makes the design matrix singular.

    Args:
        covariates_df: covariate names as index (tensorQTL file layout).
        interaction_df: interaction terms as columns.
    """
    overlap = [c for c in interaction_df.columns if c in covariates_df.index]
    if overlap:
        raise ValueError(
            f"Interaction term(s) {overlap} also present in the covariates. "
            "tensorQTL already includes interaction terms as main effects; "
            "remove them from the covariates file."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute cosinor (circadian) terms for tensorQTL interaction mapping. "
            "Writes an interaction file with cos_t and sin_t (plus cos2_t and sin2_t "
            "with --harmonics 2), and copies the covariates file after checking that "
            "it does not already contain these terms."
        )
    )
    parser.add_argument("--metadata", required=True,
                        help="Sample metadata TSV. First column = sample IDs; must contain --time-col.")
    parser.add_argument("--covariates", required=True,
                        help="Existing tensorQTL covariates file (rows=covariates, cols=samples).")
    parser.add_argument("--out-covariates", required=True,
                        help="Output path for the covariates (copied unchanged; cosinor terms are "
                             "not added because tensorQTL includes interaction terms as main effects).")
    parser.add_argument("--out-interaction", required=True,
                        help="Output path for the interaction file (samples x cos_t, sin_t[, cos2_t, sin2_t]).")
    parser.add_argument("--time-col", default="hour",
                        help="Column name for time values in metadata (default: 'hour').")
    parser.add_argument("--mode", choices=["tod", "toy"], default="tod",
                        help=(
                            "tod = time of day: encodes 24-hour circadian rhythm, "
                            "expects hours (0-24) or ISO 8601 timestamps. "
                            "toy = time of year: encodes seasonal rhythm, "
                            "expects day-of-year (1-366) or ISO 8601 timestamps. "
                            "Sets the default --period (24.0 for tod, 365.25 for toy)."
                        ))
    parser.add_argument("--period", type=float, default=None,
                        help="Cycle period (default: 24.0 for tod, 365.25 for toy). "
                             "Override with this flag if needed.")
    parser.add_argument("--harmonics", type=int, choices=[1, 2], default=1,
                        help="1 = cos_t, sin_t (24 h rhythm); 2 = also cos2_t, sin2_t "
                             "(12 h harmonic, needed for the 4df/5df tests). Default: 1.")
    args = parser.parse_args()

    if args.period is None:
        args.period = 24.0 if args.mode == "tod" else 365.25

    print(f"Mode: {args.mode}  |  Period: {args.period}")
    print(f"Loading metadata from {args.metadata}")
    hours = load_metadata(args.metadata, time_col=args.time_col, mode=args.mode)

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
    terms = compute_cosinor(hours_aligned, period=args.period, harmonics=args.harmonics)
    interaction_df = make_interaction_df(*terms)
    try:
        check_no_overlap(covariates_df, interaction_df)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    covariates_df.to_csv(args.out_covariates, sep="\t")
    print(f"Wrote covariates ({covariates_df.shape[0]} rows) to {args.out_covariates}")

    interaction_df.to_csv(args.out_interaction, sep="\t")
    print(f"Wrote interaction file ({interaction_df.shape[0]} samples x "
          f"{list(interaction_df.columns)}) to {args.out_interaction}")


if __name__ == "__main__":
    main()
