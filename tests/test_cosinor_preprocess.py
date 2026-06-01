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


from cosinor_preprocess import (
    load_metadata, load_covariates,
    append_cosinor_to_covariates, make_interaction_df,
)


def test_load_metadata_basic(tmp_path):
    f = tmp_path / "meta.tsv"
    f.write_text("sample_id\thour\nS1\t8.0\nS2\t20.5\n")
    result = load_metadata(str(f), time_col="hour")
    assert len(result) == 2
    assert result["S1"] == pytest.approx(8.0)
    assert result["S2"] == pytest.approx(20.5)


def test_load_metadata_missing_column_raises(tmp_path):
    f = tmp_path / "meta.tsv"
    f.write_text("sample_id\ttime\nS1\t8.0\n")
    with pytest.raises(ValueError, match="time_col 'hour'"):
        load_metadata(str(f), time_col="hour")


def test_load_metadata_duplicate_sample_raises(tmp_path):
    f = tmp_path / "meta.tsv"
    f.write_text("sample_id\thour\nS1\t8.0\nS1\t12.0\n")
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        load_metadata(str(f), time_col="hour")


def test_load_metadata_nonnumeric_time_raises(tmp_path):
    f = tmp_path / "meta.tsv"
    f.write_text("sample_id\thour\nS1\tnoon\n")
    with pytest.raises(ValueError):
        load_metadata(str(f), time_col="hour")


def test_load_covariates_basic(tmp_path):
    f = tmp_path / "cov.txt"
    f.write_text("ID\tS1\tS2\nPC1\t0.1\t0.2\nPC2\t-0.1\t0.3\n")
    result = load_covariates(str(f))
    assert result.shape == (2, 2)
    assert list(result.index) == ["PC1", "PC2"]
    assert list(result.columns) == ["S1", "S2"]


def test_append_cosinor_to_covariates_adds_rows():
    cov = pd.DataFrame({"S1": [0.1], "S2": [0.2]}, index=["PC1"])
    cos_t = pd.Series({"S1": 0.866, "S2": 0.5})
    sin_t = pd.Series({"S1": 0.5, "S2": 0.866})
    result = append_cosinor_to_covariates(cov, cos_t, sin_t)
    assert list(result.index) == ["PC1", "cos_t", "sin_t"]
    assert result.loc["cos_t", "S1"] == pytest.approx(0.866)
    assert result.loc["sin_t", "S2"] == pytest.approx(0.866)


def test_append_cosinor_raises_if_cos_t_already_present():
    cov = pd.DataFrame({"S1": [0.1]}, index=["cos_t"])
    cos_t = pd.Series({"S1": 0.866})
    sin_t = pd.Series({"S1": 0.5})
    with pytest.raises(ValueError, match="already present"):
        append_cosinor_to_covariates(cov, cos_t, sin_t)


def test_append_cosinor_raises_if_sin_t_already_present():
    cov = pd.DataFrame({"S1": [0.1]}, index=["sin_t"])
    cos_t = pd.Series({"S1": 0.866})
    sin_t = pd.Series({"S1": 0.5})
    with pytest.raises(ValueError, match="already present"):
        append_cosinor_to_covariates(cov, cos_t, sin_t)


def test_make_interaction_df_shape_and_name():
    cos_t = pd.Series({"S1": 0.866, "S2": 0.5})
    result = make_interaction_df(cos_t)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 1)
    assert result.columns[0] == "cos_t"
    assert list(result.index) == ["S1", "S2"]


def test_append_cosinor_does_not_mutate_input():
    cov = pd.DataFrame({"S1": [0.1]}, index=["PC1"])
    cov_orig = cov.copy()
    cos_t = pd.Series({"S1": 0.866})
    sin_t = pd.Series({"S1": 0.5})
    append_cosinor_to_covariates(cov, cos_t, sin_t)
    pd.testing.assert_frame_equal(cov, cov_orig)


import subprocess


def test_main_cli_end_to_end(tmp_path):
    meta = tmp_path / "meta.tsv"
    meta.write_text(
        "sample_id\thour\n"
        "S1\t0.0\n"
        "S2\t6.0\n"
        "S3\t12.0\n"
    )
    cov = tmp_path / "cov.txt"
    cov.write_text("ID\tS1\tS2\tS3\nPC1\t0.1\t0.2\t0.3\n")
    out_cov = tmp_path / "out_cov.txt"
    out_int = tmp_path / "out_int.txt"

    result = subprocess.run(
        [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "..", "scripts", "cosinor_preprocess.py"),
            "--metadata", str(meta),
            "--covariates", str(cov),
            "--out-covariates", str(out_cov),
            "--out-interaction", str(out_int),
            "--time-col", "hour",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    out_cov_df = pd.read_csv(str(out_cov), sep="\t", index_col=0)
    assert list(out_cov_df.index) == ["PC1", "cos_t", "sin_t"]
    assert abs(out_cov_df.loc["cos_t", "S1"] - 1.0) < 1e-6    # cos(0) = 1
    assert abs(out_cov_df.loc["cos_t", "S2"] - 0.0) < 1e-6    # cos(π/2) = 0
    assert abs(out_cov_df.loc["cos_t", "S3"] - (-1.0)) < 1e-6  # cos(π) = -1

    out_int_df = pd.read_csv(str(out_int), sep="\t", index_col=0)
    assert list(out_int_df.columns) == ["cos_t"]
    assert list(out_int_df.index) == ["S1", "S2", "S3"]


def test_main_cli_missing_samples_raises(tmp_path):
    # Metadata has S1, S2 but covariates has S1, S2, S3 → S3 missing
    meta = tmp_path / "meta.tsv"
    meta.write_text("sample_id\thour\nS1\t0.0\nS2\t6.0\n")
    cov = tmp_path / "cov.txt"
    cov.write_text("ID\tS1\tS2\tS3\nPC1\t0.1\t0.2\t0.3\n")
    out_cov = tmp_path / "out_cov.txt"
    out_int = tmp_path / "out_int.txt"

    result = subprocess.run(
        [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "..", "scripts", "cosinor_preprocess.py"),
            "--metadata", str(meta),
            "--covariates", str(cov),
            "--out-covariates", str(out_cov),
            "--out-interaction", str(out_int),
            "--time-col", "hour",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "S3" in result.stderr or "S3" in result.stdout
