import pytest
import pandas as pd
from pathlib import Path
import torch

# Assuming datasets.py is in protein_rna_3d package relative to project root
# Adjust import path if necessary
try:
    from protein_rna_3d.datasets import RNA3DDataset
except ImportError:
    pytest.skip("RNA3DDataset not found, skipping tests", allow_module_level=True)

# --- Test Data Fixtures ---


@pytest.fixture(scope="module")
def mock_data_dir(tmp_path_factory):
    """Creates a temporary directory with mock data files."""
    data_dir = tmp_path_factory.mktemp("mock_rna_data")

    # Create train files
    train_seq_data = {
        "target_id": ["seqA", "seqB", "seqC", "seqD_mismatch"],
        "sequence": ["GCA", "AUUGC", "G", "AG"],  # seqB len=5, seqD len=2
        "temporal_cutoff": ["2020-01-01"] * 4,
        "description": ["desc"] * 4,
        "all_sequences": ["fasta"] * 4,
    }
    pd.DataFrame(train_seq_data).to_csv(data_dir / "train_sequences.csv", index=False)

    train_label_data = {
        "ID": [
            "seqA_1",
            "seqA_2",
            "seqA_3",
            "seqD_mismatch_1",
            "seqD_mismatch_2",
            "seqD_mismatch_3",
        ],  # seqD len=3
        "resname": ["G", "C", "A", "A", "G", "G"],
        "resid": [1, 2, 3, 1, 2, 3],
        "x_1": [1.0, 2.0, 3.0, 10.0, 11.0, 12.0],
        "y_1": [1.1, 2.1, 3.1, 10.1, 11.1, 12.1],
        "z_1": [1.2, 2.2, 3.2, 10.2, 11.2, 12.2],
        # No labels for seqB or seqC
    }
    pd.DataFrame(train_label_data).to_csv(data_dir / "train_labels.csv", index=False)

    # Create test files (no labels)
    test_seq_data = {
        "target_id": ["seqT1", "seqT2"],
        "sequence": ["UU", "ACGA"],
        "temporal_cutoff": ["2023-01-01"] * 2,
        "description": ["desc"] * 2,
        "all_sequences": ["fasta"] * 2,
    }
    pd.DataFrame(test_seq_data).to_csv(data_dir / "test_sequences.csv", index=False)

    # Create validation files (can reuse train for simplicity here if needed, or make specific)
    val_seq_data = {
        "target_id": ["seqV1"],
        "sequence": ["C"],
        "temporal_cutoff": ["2021-01-01"],
        "description": ["desc"],
        "all_sequences": ["fasta"],
    }
    pd.DataFrame(val_seq_data).to_csv(
        data_dir / "validation_sequences.csv", index=False
    )
    # No validation_labels.csv for this simple test case

    return data_dir


# --- Dataset Initialization Tests ---


def test_dataset_init_train_success(mock_data_dir):
    """Test successful initialization with train split."""
    dataset = RNA3DDataset(data_dir=mock_data_dir, split="train")
    assert len(dataset) == 4
    assert dataset.split == "train"
    assert dataset.labels_df is not None
    assert "seqA" in dataset.target_id_to_coords
    assert "seqD_mismatch" in dataset.target_id_to_coords
    assert "seqB" not in dataset.target_id_to_coords  # Missing from labels file
    assert "seqC" not in dataset.target_id_to_coords  # Missing from labels file


def test_dataset_init_test_success(mock_data_dir):
    """Test successful initialization with test split (no labels)."""
    dataset = RNA3DDataset(data_dir=mock_data_dir, split="test")
    assert len(dataset) == 2
    assert dataset.split == "test"
    assert dataset.labels_df is None
    assert (
        not hasattr(dataset, "target_id_to_coords") or not dataset.target_id_to_coords
    )


def test_dataset_init_validation_no_labels(mock_data_dir, capsys):
    """Test initialization with validation split when labels are missing."""
    dataset = RNA3DDataset(data_dir=mock_data_dir, split="validation")
    assert len(dataset) == 1
    assert dataset.split == "validation"
    assert dataset.labels_df is None  # Should be None as file doesn't exist
    captured = capsys.readouterr()
    assert "Warning: Label file not found for split 'validation'" in captured.out


def test_dataset_init_invalid_split(mock_data_dir):
    """Test initialization fails with an invalid split name."""
    with pytest.raises(ValueError, match="Invalid split 'invalid_split'"):
        RNA3DDataset(data_dir=mock_data_dir, split="invalid_split")


def test_dataset_init_missing_sequence_file(tmp_path):
    """Test initialization fails if sequence file is missing."""
    with pytest.raises(FileNotFoundError, match="train_sequences.csv"):
        RNA3DDataset(data_dir=tmp_path, split="train")


# --- Dataset Length Test ---


def test_dataset_len(mock_data_dir):
    """Test __len__ method."""
    train_dataset = RNA3DDataset(data_dir=mock_data_dir, split="train")
    test_dataset = RNA3DDataset(data_dir=mock_data_dir, split="test")
    assert len(train_dataset) == 4
    assert len(test_dataset) == 2


# --- Dataset Get Item Tests ---


def test_dataset_getitem_train_success(mock_data_dir):
    """Test __getitem__ for a sample with sequence and labels."""
    dataset = RNA3DDataset(data_dir=mock_data_dir, split="train")
    sample = dataset[0]  # seqA

    assert isinstance(sample, dict)
    assert sample["target_id"] == "seqA"
    assert sample["sequence"] == "GCA"
    assert "coordinates" in sample
    coords = sample["coordinates"]
    assert isinstance(coords, torch.Tensor)
    assert coords.shape == (3, 3)  # 3 residues, 3 coords (x, y, z)
    assert coords.dtype == torch.float32
    # Check first coordinate values
    assert torch.allclose(coords[0], torch.tensor([1.0, 1.1, 1.2]))


def test_dataset_getitem_no_labels_available(mock_data_dir):
    """Test __getitem__ for a sequence present but without labels (seqB/seqC)."""
    dataset = RNA3DDataset(data_dir=mock_data_dir, split="train")
    sample_b = dataset[1]  # seqB
    sample_c = dataset[2]  # seqC

    assert sample_b["target_id"] == "seqB"
    assert sample_b["sequence"] == "AUUGC"
    assert "coordinates" not in sample_b

    assert sample_c["target_id"] == "seqC"
    assert sample_c["sequence"] == "G"
    assert "coordinates" not in sample_c


def test_dataset_getitem_length_mismatch(mock_data_dir, capsys):
    """Test __getitem__ where sequence length mismatches coordinate length."""
    dataset = RNA3DDataset(data_dir=mock_data_dir, split="train")
    sample = dataset[3]  # seqD_mismatch (seq len 2, coord len 3)

    assert sample["target_id"] == "seqD_mismatch"
    assert sample["sequence"] == "AG"
    # Coordinates should be skipped due to mismatch
    assert "coordinates" not in sample

    # Check warning message
    captured = capsys.readouterr()
    assert (
        "Warning: Sequence length 2 mismatch with coordinates length 3" in captured.out
    )
    assert "target seqD_mismatch" in captured.out


def test_dataset_getitem_test_split(mock_data_dir):
    """Test __getitem__ for the test split (no coordinates expected)."""
    dataset = RNA3DDataset(data_dir=mock_data_dir, split="test")
    sample = dataset[0]  # seqT1

    assert isinstance(sample, dict)
    assert sample["target_id"] == "seqT1"
    assert sample["sequence"] == "UU"
    assert "coordinates" not in sample


# --- Test Label Preprocessing Edge Cases (Implicitly via init) ---


def test_preprocess_labels_malformed_id(tmp_path):
    """Test label preprocessing handles IDs that don't match expected format."""
    data_dir = tmp_path
    seq_data = {
        "target_id": ["seqA"],
        "sequence": ["GCA"],
        "temporal_cutoff": ["d"],
        "description": ["d"],
        "all_sequences": ["f"],
    }
    pd.DataFrame(seq_data).to_csv(data_dir / "train_sequences.csv", index=False)

    # Label ID doesn't end in _<number>
    label_data = {
        "ID": ["seqA"],
        "resname": ["G"],
        "resid": [1],
        "x_1": [1],
        "y_1": [1],
        "z_1": [1],
    }
    pd.DataFrame(label_data).to_csv(data_dir / "train_labels.csv", index=False)

    dataset = RNA3DDataset(data_dir=data_dir, split="train")
    # Expect error during preprocessing, labels should be invalidated
    assert dataset.labels_df is None
    assert (
        not hasattr(dataset, "target_id_to_coords") or not dataset.target_id_to_coords
    )
    # We could also capture stderr here to check for the specific error message
