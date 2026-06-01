import pytest
import math
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from cosinor_preprocess import parse_time_to_hours, compute_cosinor


def test_placeholder():
    pass


def test_parse_time_to_hours_numeric_string():
    assert parse_time_to_hours("12.5") == 12.5


def test_parse_time_to_hours_integer_string():
    assert parse_time_to_hours("0") == 0.0


def test_parse_time_to_hours_nonnumeric_raises():
    with pytest.raises(ValueError):
        parse_time_to_hours("not_a_number")


def test_compute_cosinor_known_values():
    hours = pd.Series({"S1": 0.0, "S2": 6.0, "S3": 12.0, "S4": 18.0})
    cos_t, sin_t = compute_cosinor(hours, period=24.0)
    assert abs(cos_t["S1"] - 1.0) < 1e-6    # cos(0) = 1
    assert abs(cos_t["S2"] - 0.0) < 1e-6    # cos(π/2) = 0
    assert abs(cos_t["S3"] - (-1.0)) < 1e-6  # cos(π) = -1
    assert abs(cos_t["S4"] - 0.0) < 1e-6    # cos(3π/2) ≈ 0
    assert abs(sin_t["S1"] - 0.0) < 1e-6    # sin(0) = 0
    assert abs(sin_t["S2"] - 1.0) < 1e-6    # sin(π/2) = 1
    assert abs(sin_t["S3"]) < 1e-6          # sin(π) ≈ 0 (floating point)
    assert abs(sin_t["S4"] - (-1.0)) < 1e-6  # sin(3π/2) = -1


def test_compute_cosinor_preserves_index():
    hours = pd.Series({"A": 3.0, "B": 9.0})
    cos_t, sin_t = compute_cosinor(hours)
    assert list(cos_t.index) == ["A", "B"]
    assert list(sin_t.index) == ["A", "B"]


def test_compute_cosinor_series_names():
    hours = pd.Series({"S1": 6.0})
    cos_t, sin_t = compute_cosinor(hours)
    assert cos_t.name == "cos_t"
    assert sin_t.name == "sin_t"


def test_compute_cosinor_custom_period():
    # 12-hour period: hour=6 → angle=π → cos=-1
    hours = pd.Series({"S1": 6.0})
    cos_t, _ = compute_cosinor(hours, period=12.0)
    assert abs(cos_t["S1"] - (-1.0)) < 1e-6
