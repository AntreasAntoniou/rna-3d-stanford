import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# --- Setup: Add scripts directory to sys.path ---
SCRIPT_DIR = Path(__file__).parent.parent / "scripts"
if not SCRIPT_DIR.is_dir():
    pytest.skip("Scripts directory not found", allow_module_level=True)
sys.path.insert(
    0, str(SCRIPT_DIR.parent)
)  # Add project root to allow 'from scripts import ...'

# --- Import the script module ---
try:
    # Import functions directly if the script is importable
    from scripts.create_submission import (
        create_submission,
        load_predictions,
        main as create_submission_main,  # Avoid name clash with test function
    )
except ImportError as e:
    pytest.skip(
        f"Could not import create_submission script: {e}", allow_module_level=True
    )

# --- Test Data Paths ---
# Assuming test data files exist in tests/data relative to this test file
TEST_DATA_ROOT = Path(__file__).parent / "data"
VALID_SEQS_CSV = TEST_DATA_ROOT / "test_sequences.csv"
VALID_PREDS_JSON = TEST_DATA_ROOT / "test_predictions.json"
BAD_FORMAT_PREDS_JSON = TEST_DATA_ROOT / "bad_format_predictions.json"
MISMATCHED_LEN_PREDS_JSON = TEST_DATA_ROOT / "mismatched_len_predictions.json"
NON_NUMERIC_PREDS_JSON = TEST_DATA_ROOT / "non_numeric_predictions.json"

# --- Pytest Fixtures ---


@pytest.fixture(scope="module")
def check_test_data_files():
    """Ensure necessary test data files exist before running tests."""
    required_files = [
        VALID_SEQS_CSV,
        VALID_PREDS_JSON,
        BAD_FORMAT_PREDS_JSON,
        MISMATCHED_LEN_PREDS_JSON,
        NON_NUMERIC_PREDS_JSON,
    ]
    missing = [f for f in required_files if not f.is_file()]
    if missing:
        pytest.skip(
            f"Missing test data files: {', '.join(map(str, missing))}",
            allow_module_level=True,
        )


@pytest.fixture
def temp_output_csv(tmp_path):
    """Provides a path for a temporary output CSV file."""
    return tmp_path / "submission_test.csv"


# --- Test load_predictions Function ---


@pytest.mark.usefixtures("check_test_data_files")
def test_load_predictions_success():
    """Test loading a valid predictions JSON file."""
    predictions = load_predictions(str(VALID_PREDS_JSON))

    assert isinstance(predictions, dict)
    assert "seq_001" in predictions
    assert "seq_002" in predictions
    assert len(predictions["seq_001"]) == 5
    assert all(isinstance(p, np.ndarray) for p in predictions["seq_001"])
    assert all(p.shape == (3, 3) for p in predictions["seq_001"])
    assert predictions["seq_001"][0][0, 0] == pytest.approx(1.0)
    assert predictions["seq_001"][0].dtype == float


def test_load_predictions_file_not_found(tmp_path):
    """Test loading with a non-existent file should exit."""
    non_existent_path = tmp_path / "non_existent.json"
    with pytest.raises(SystemExit) as e:
        load_predictions(str(non_existent_path))
    assert e.value.code == 1


def test_load_predictions_bad_json(tmp_path):
    """Test loading invalid JSON should exit."""
    bad_json_path = tmp_path / "bad.json"
    bad_json_path.write_text('{"key": [1, 2,')  # Malformed JSON
    with pytest.raises(SystemExit) as e:
        load_predictions(str(bad_json_path))
    assert e.value.code == 1


def test_load_predictions_wrong_top_type(tmp_path):
    """Test loading JSON that is not a dictionary/object should exit."""
    wrong_type_path = tmp_path / "wrong_type.json"
    wrong_type_path.write_text("[1, 2, 3]")  # JSON array, not object
    with pytest.raises(SystemExit) as e:
        load_predictions(str(wrong_type_path))
    assert e.value.code == 1


@pytest.mark.usefixtures("check_test_data_files")
def test_load_predictions_bad_format():
    """Test loading JSON with incorrect list structure (not 5 preds) should exit."""
    with pytest.raises(SystemExit) as e:
        load_predictions(str(BAD_FORMAT_PREDS_JSON))
    assert e.value.code == 1


@pytest.mark.usefixtures("check_test_data_files")
def test_load_predictions_mismatched_length():
    """Test loading JSON where prediction lists have different lengths should exit."""
    with pytest.raises(SystemExit) as e:
        load_predictions(str(MISMATCHED_LEN_PREDS_JSON))
    assert e.value.code == 1


@pytest.mark.usefixtures("check_test_data_files")
def test_load_predictions_non_numeric():
    """Test loading JSON with non-numeric coordinate values should exit."""
    with pytest.raises(SystemExit) as e:
        load_predictions(str(NON_NUMERIC_PREDS_JSON))
    assert e.value.code == 1


def test_load_predictions_wrong_coord_shape(tmp_path):
    """Test loading JSON where coords are not (L, 3) should exit."""
    wrong_shape_path = tmp_path / "wrong_shape.json"
    # Shape (3, 2) instead of (3, 3)
    wrong_shape_json = {
        "seq_001": [
            [[1, 2], [3, 4], [5, 6]],
            [[1, 2], [3, 4], [5, 6]],
            [[1, 2], [3, 4], [5, 6]],
            [[1, 2], [3, 4], [5, 6]],
            [[1, 2], [3, 4], [5, 6]],
        ]
    }
    wrong_shape_path.write_text(json.dumps(wrong_shape_json))
    with pytest.raises(SystemExit) as e:
        load_predictions(str(wrong_shape_path))
    assert e.value.code == 1


# --- Test create_submission Function ---


@pytest.mark.usefixtures("check_test_data_files")
def test_create_submission_success(temp_output_csv):
    """Test successful creation of submission.csv from valid inputs."""
    predictions = load_predictions(str(VALID_PREDS_JSON))
    sequences_df = pd.read_csv(VALID_SEQS_CSV)

    create_submission(predictions, sequences_df, str(temp_output_csv))

    assert temp_output_csv.exists()
    submission_df = pd.read_csv(temp_output_csv)

    # Expected rows: 3 from seq_001 + 3 from seq_002 = 6
    assert len(submission_df) == 6
    # Check columns (order matters)
    expected_cols = ["ID", "resname", "resid"] + [
        f"{c}_{i}" for i in range(1, 6) for c in ["x", "y", "z"]
    ]
    assert list(submission_df.columns) == expected_cols

    # Check specific values
    assert submission_df.loc[0, "ID"] == "seq_001_1"
    assert submission_df.loc[0, "resname"] == "G"
    assert submission_df.loc[0, "resid"] == 1
    assert submission_df.loc[0, "x_1"] == pytest.approx(1.0)
    assert submission_df.loc[0, "y_1"] == pytest.approx(1.1)
    assert submission_df.loc[0, "z_1"] == pytest.approx(1.2)
    assert submission_df.loc[0, "x_5"] == pytest.approx(5.0)
    assert submission_df.loc[0, "z_5"] == pytest.approx(5.2)

    assert submission_df.loc[3, "ID"] == "seq_002_1"
    assert submission_df.loc[3, "resname"] == "C"
    assert submission_df.loc[3, "x_1"] == pytest.approx(10.0)
    assert submission_df.loc[5, "ID"] == "seq_002_3"
    assert submission_df.loc[5, "z_5"] == pytest.approx(14.8)


@pytest.mark.usefixtures("check_test_data_files")
def test_create_submission_missing_prediction(temp_output_csv, capsys):
    """Test creates file but warns when a sequence_id has no prediction."""
    # seq_003 is in sequences CSV but not predictions JSON
    predictions = load_predictions(str(VALID_PREDS_JSON))
    sequences_df = pd.read_csv(VALID_SEQS_CSV)

    create_submission(predictions, sequences_df, str(temp_output_csv))

    captured = capsys.readouterr()
    assert "Warning: No predictions found for sequence seq_003" in captured.err

    # Submission file should still be created, containing only seq_001 and seq_002
    assert temp_output_csv.exists()
    submission_df = pd.read_csv(temp_output_csv)
    assert len(submission_df) == 6  # 3 rows for seq_001 + 3 for seq_002


@pytest.mark.usefixtures("check_test_data_files")
def test_create_submission_length_mismatch(temp_output_csv, capsys):
    """Test creates file but warns and skips seqs where pred length != seq length."""
    # Create predictions where seq_001 has length 2 instead of 3
    preds_dict_bad_len = load_predictions(str(VALID_PREDS_JSON))
    preds_dict_bad_len["seq_001"] = [
        p[:2] for p in preds_dict_bad_len["seq_001"]
    ]  # Slice to length 2

    sequences_df = pd.read_csv(VALID_SEQS_CSV)

    create_submission(preds_dict_bad_len, sequences_df, str(temp_output_csv))

    captured = capsys.readouterr()
    assert (
        "Warning: Prediction length (2) does not match sequence length (3) for sequence seq_001"
        in captured.err
    )

    # Submission file should contain only seq_002 (3 rows)
    assert temp_output_csv.exists()
    submission_df = pd.read_csv(temp_output_csv)
    assert len(submission_df) == 3
    assert submission_df["ID"].str.startswith("seq_002").all()


@pytest.mark.usefixtures("check_test_data_files")
def test_create_submission_no_valid_data(temp_output_csv):
    """Test script exits if no valid submission rows can be generated."""
    predictions = {}  # Empty predictions
    sequences_df = pd.read_csv(VALID_SEQS_CSV)

    with pytest.raises(SystemExit) as e:
        create_submission(predictions, sequences_df, str(temp_output_csv))
    assert e.value.code == 1
    assert not temp_output_csv.exists()  # Should exit before creating the file


# --- Test main Script Execution (using subprocess) ---


@pytest.mark.usefixtures("check_test_data_files")
def test_script_execution_success(temp_output_csv):
    """Test running the script successfully via command line."""
    script_path = SCRIPT_DIR / "create_submission.py"
    cmd = [
        sys.executable,  # Use the same python running pytest
        str(script_path),
        "--predictions",
        str(VALID_PREDS_JSON),
        "--sequences",
        str(VALID_SEQS_CSV),
        "--output",
        str(temp_output_csv),
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, check=True, encoding="utf-8"
    )

    assert temp_output_csv.exists()
    assert "Successfully created submission file" in result.stdout
    submission_df = pd.read_csv(temp_output_csv)
    assert len(submission_df) == 6  # seq_001 + seq_002 data


@pytest.mark.usefixtures("check_test_data_files")
def test_script_execution_missing_args():
    """Test script fails with missing required arguments."""
    script_path = SCRIPT_DIR / "create_submission.py"
    cmd_missing_preds = [
        sys.executable,
        str(script_path),
        "--sequences",
        str(VALID_SEQS_CSV),
    ]
    cmd_missing_seqs = [
        sys.executable,
        str(script_path),
        "--predictions",
        str(VALID_PREDS_JSON),
    ]

    result_missing_preds = subprocess.run(
        cmd_missing_preds, capture_output=True, text=True, encoding="utf-8"
    )
    result_missing_seqs = subprocess.run(
        cmd_missing_seqs, capture_output=True, text=True, encoding="utf-8"
    )

    assert result_missing_preds.returncode != 0
    assert (
        "the following arguments are required: --predictions"
        in result_missing_preds.stderr
    )
    assert result_missing_seqs.returncode != 0
    assert (
        "the following arguments are required: --sequences"
        in result_missing_seqs.stderr
    )


@pytest.mark.usefixtures("check_test_data_files")
def test_script_execution_bad_input_files(tmp_path):
    """Test script fails gracefully with bad input file paths."""
    script_path = SCRIPT_DIR / "create_submission.py"
    output_csv = tmp_path / "submission_fail.csv"

    # Bad predictions path
    cmd_bad_preds = [
        sys.executable,
        str(script_path),
        "--predictions",
        str(tmp_path / "non_existent.json"),
        "--sequences",
        str(VALID_SEQS_CSV),
        "--output",
        str(output_csv),
    ]
    result_bad_preds = subprocess.run(
        cmd_bad_preds, capture_output=True, text=True, encoding="utf-8"
    )
    assert result_bad_preds.returncode != 0
    assert "Predictions file not found" in result_bad_preds.stderr

    # Bad sequences path
    cmd_bad_seqs = [
        sys.executable,
        str(script_path),
        "--predictions",
        str(VALID_PREDS_JSON),
        "--sequences",
        str(tmp_path / "non_existent.csv"),
        "--output",
        str(output_csv),
    ]
    result_bad_seqs = subprocess.run(
        cmd_bad_seqs, capture_output=True, text=True, encoding="utf-8"
    )
    assert result_bad_seqs.returncode != 0
    assert "Sequence file not found" in result_bad_seqs.stderr
