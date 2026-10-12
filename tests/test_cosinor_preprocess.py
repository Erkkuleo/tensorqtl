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
    make_interaction_df, check_no_overlap,
    parse_time_to_day_of_year,
)


def test_parse_time_to_day_of_year_numeric():
    assert parse_time_to_day_of_year("180.5") == pytest.approx(180.5)


def test_parse_time_to_day_of_year_iso():
    # 2024-01-01 is day 1
    assert parse_time_to_day_of_year("2024-01-01T00:00:00") == pytest.approx(1.0)
    # 2024-07-01 is day 183 (2024 is a leap year)
    assert parse_time_to_day_of_year("2024-07-01T12:00:00") == pytest.approx(183.0)


def test_parse_time_to_day_of_year_invalid():
    with pytest.raises(ValueError):
        parse_time_to_day_of_year("not_a_date")


def test_load_metadata_basic(tmp_path):
    f = tmp_path / "meta.tsv"
    f.write_text("sample_id\thour\nS1\t8.0\nS2\t20.5\n")
    result = load_metadata(str(f), time_col="hour")
    assert len(result) == 2
    assert result["S1"] == pytest.approx(8.0)
    assert result["S2"] == pytest.approx(20.5)


def test_load_metadata_toy_mode(tmp_path):
    f = tmp_path / "meta.tsv"
    f.write_text("sample_id\tday\nS1\t1.0\nS2\t180.0\n")
    result = load_metadata(str(f), time_col="day", mode="toy")
    assert result["S1"] == pytest.approx(1.0)
    assert result["S2"] == pytest.approx(180.0)


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


def test_make_interaction_df_shape_and_name():
    cos_t = pd.Series({"S1": 0.866, "S2": 0.5})
    sin_t = pd.Series({"S1": 0.5, "S2": 0.866})
    result = make_interaction_df(cos_t, sin_t)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 2)
    assert list(result.columns) == ["cos_t", "sin_t"]
    assert list(result.index) == ["S1", "S2"]


def test_compute_cosinor_two_harmonics():
    hours = pd.Series({"S1": 0.0, "S2": 3.0, "S3": 6.0})
    terms = compute_cosinor(hours, period=24.0, harmonics=2)
    assert [t.name for t in terms] == ["cos_t", "sin_t", "cos2_t", "sin2_t"]
    cos2_t, sin2_t = terms[2], terms[3]
    assert cos2_t["S1"] == pytest.approx(1.0)
    assert sin2_t["S2"] == pytest.approx(1.0)                 # 12 h harmonic: 3 h -> pi/2
    assert cos2_t["S3"] == pytest.approx(-1.0)                # 6 h -> pi
    assert terms[0]["S3"] == pytest.approx(0.0, abs=1e-12)    # fundamental unchanged


def test_compute_cosinor_invalid_harmonics():
    with pytest.raises(ValueError):
        compute_cosinor(pd.Series({"S1": 1.0}), harmonics=3)


def test_make_interaction_df_four_columns():
    hours = pd.Series({"S1": 0.0, "S2": 7.0})
    result = make_interaction_df(*compute_cosinor(hours, harmonics=2))
    assert list(result.columns) == ["cos_t", "sin_t", "cos2_t", "sin2_t"]
    assert list(result.index) == ["S1", "S2"]


def test_check_no_overlap_ok():
    cov = pd.DataFrame({"S1": [0.1]}, index=["PC1"])
    inter = pd.DataFrame({"cos_t": [1.0], "sin_t": [0.0]}, index=["S1"])
    check_no_overlap(cov, inter)  # must not raise


def test_check_no_overlap_raises():
    cov = pd.DataFrame({"S1": [0.1, 0.5]}, index=["PC1", "sin_t"])
    inter = pd.DataFrame({"cos_t": [1.0], "sin_t": [0.0]}, index=["S1"])
    with pytest.raises(ValueError, match="sin_t"):
        check_no_overlap(cov, inter)


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

    # cosinor terms are NOT added to the covariates (they enter as interaction main effects)
    out_cov_df = pd.read_csv(str(out_cov), sep="\t", index_col=0)
    assert list(out_cov_df.index) == ["PC1"]

    out_int_df = pd.read_csv(str(out_int), sep="\t", index_col=0)
    assert list(out_int_df.columns) == ["cos_t", "sin_t"]
    assert list(out_int_df.index) == ["S1", "S2", "S3"]
    assert abs(out_int_df.loc["S1", "cos_t"] - 1.0) < 1e-6    # cos(0) = 1
    assert abs(out_int_df.loc["S2", "cos_t"] - 0.0) < 1e-6    # cos(π/2) = 0
    assert abs(out_int_df.loc["S3", "cos_t"] - (-1.0)) < 1e-6  # cos(π) = -1


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


def _run_cli(tmp_path, cov_text, extra_args=()):
    meta = tmp_path / "meta.tsv"
    meta.write_text("sample_id\thour\nS1\t0.0\nS2\t6.0\nS3\t12.0\n")
    cov = tmp_path / "cov.txt"
    cov.write_text(cov_text)
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
            *extra_args,
        ],
        capture_output=True, text=True,
    )
    return result, out_int


def test_main_cli_harmonics_2(tmp_path):
    result, out_int = _run_cli(tmp_path, "ID\tS1\tS2\tS3\nPC1\t0.1\t0.2\t0.3\n",
                               ["--harmonics", "2"])
    assert result.returncode == 0, result.stderr
    out_int_df = pd.read_csv(str(out_int), sep="\t", index_col=0)
    assert list(out_int_df.columns) == ["cos_t", "sin_t", "cos2_t", "sin2_t"]
    assert abs(out_int_df.loc["S2", "cos2_t"] - (-1.0)) < 1e-6  # 6 h -> pi for the 12 h harmonic


def test_main_cli_cosinor_term_in_covariates_raises(tmp_path):
    result, _ = _run_cli(tmp_path, "ID\tS1\tS2\tS3\nPC1\t0.1\t0.2\t0.3\ncos_t\t1\t0\t-1\n")
    assert result.returncode != 0
    assert "cos_t" in result.stderr


def test_main_cli_toy_mode_harmonics_2(tmp_path):
    meta = tmp_path / "meta.tsv"
    meta.write_text("sample_id\tday\nS1\t1.0\nS2\t180.0\n")
    cov = tmp_path / "cov.txt"
    cov.write_text("ID\tS1\tS2\nPC1\t0.1\t0.2\n")
    out_int = tmp_path / "out_int.txt"
    result = subprocess.run(
        [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "..", "scripts", "cosinor_preprocess.py"),
            "--metadata", str(meta), "--covariates", str(cov),
            "--out-covariates", str(tmp_path / "out_cov.txt"),
            "--out-interaction", str(out_int),
            "--time-col", "day", "--mode", "toy", "--harmonics", "2",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    out_int_df = pd.read_csv(str(out_int), sep="\t", index_col=0)
    assert list(out_int_df.columns) == ["cos_t", "sin_t", "cos2_t", "sin2_t"]
