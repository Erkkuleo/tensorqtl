#!/usr/bin/env python3
"""Cosinor preprocessing for circadian eQTL mapping."""
import math
import numpy as np
import pandas as pd


def parse_time_to_hours(value: str) -> float:
    """Convert a time value string to fractional hours (0-24).

    STUB: currently casts to float. To support ISO 8601 timestamps
    (e.g., "2024-01-15T14:30:00"), replace the body of this function
    with ISO 8601 parsing. All downstream code consumes the returned
    float, so no other changes are needed.

    Replacement example:
        from datetime import datetime
        dt = datetime.fromisoformat(value)
        return dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        raise ValueError(
            f"Cannot convert '{value}' to hours. Expected a numeric value (e.g. '14.5')."
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
