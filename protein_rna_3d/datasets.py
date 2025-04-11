import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import argparse

# Attempt to import parser from data_utils
try:
    from .data_utils import parse_fasta_msa
except ImportError:
    # Fallback if running script directly or structure issues
    try:
        from data_utils import parse_fasta_msa
    except ImportError:
        # Define a dummy parser if import fails, so the class definition doesn't break
        print("Warning: Could not import parse_fasta_msa. MSA loading will be skipped.")

        def parse_fasta_msa(fasta_path):
            return []


class RNA3DDataset(Dataset):
    """PyTorch Dataset for RNA sequence to 3D coordinate prediction."""

    def __init__(self, data_dir: str | Path, split: str = "train", transform=None):
        """
        Args:
            data_dir (str or Path): Path to the directory containing competition data
                                    (e.g., 'data/stanford-rna-3d-folding/').
            split (str): Which data split to load ('train', 'validation', or 'test').
                         Currently only 'train' is fully implemented for labels.
            transform (callable, optional): Optional transform to be applied
                                           on a sample.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.msa_dir = self.data_dir / "MSA"  # Store MSA directory path

        allowed_splits = ["train", "validation", "test"]
        if split not in allowed_splits:
            raise ValueError(
                f"Invalid split '{split}'. Must be one of {allowed_splits}"
            )

        # Load sequences
        sequences_path = self.data_dir / f"{split}_sequences.csv"
        if not sequences_path.exists():
            raise FileNotFoundError(f"Sequence file not found: {sequences_path}")
        self.sequences_df = pd.read_csv(sequences_path)

        # Load labels (only for train/validation)
        self.labels_df = None
        self.target_id_to_coords = {}  # Initialize always
        if split in ["train", "validation"]:
            labels_path = self.data_dir / f"{split}_labels.csv"
            if not labels_path.exists():
                print(
                    f"Warning: Label file not found for split '{split}': {labels_path}"
                )
            else:
                self.labels_df = pd.read_csv(labels_path)
                # Preprocess labels: Pivot to have target_id as index
                self._preprocess_labels()

    def _preprocess_labels(self):
        """Processes the labels DataFrame for easier lookup."""
        if self.labels_df is None:
            return

        # Extract target_id from the ID column (e.g., "pdb_id_chain_id_resid")
        # Note: This assumes standard format. Might need adjustment if IDs vary.
        try:
            # Ensure ID column is string type before splitting
            self.labels_df["ID"] = self.labels_df["ID"].astype(str)
            self.labels_df[["target_id", "resid_str"]] = self.labels_df[
                "ID"
            ].str.rsplit("_", n=1, expand=True)

            # Convert resid to numeric, coercing errors
            self.labels_df["resid"] = pd.to_numeric(
                self.labels_df["resid_str"], errors="coerce"
            )
            # Drop rows where resid could not be parsed or was missing
            self.labels_df.dropna(subset=["target_id", "resid"], inplace=True)
            self.labels_df["resid"] = self.labels_df["resid"].astype(int)

            # Keep only essential columns for coordinate extraction
            # For training, we primarily use the first structure (_1)
            coord_cols = ["target_id", "resid", "x_1", "y_1", "z_1"]
            # Check if columns exist, handle cases with multiple structures if needed later
            available_coord_cols = [
                c for c in coord_cols if c in self.labels_df.columns
            ]
            coord_data_cols = [
                c for c in available_coord_cols if c not in ["target_id", "resid"]
            ]

            if (
                len(coord_data_cols) < 3 and self.split == "train"
            ):  # Expect x_1,y_1,z_1 for train
                print(
                    f"Warning: Missing expected coordinate columns (x_1, y_1, z_1) in {self.split}_labels.csv"
                )
                # Continue, but coordinates may be empty

            # Sort by target_id and resid to ensure correct order
            self.labels_df = self.labels_df[available_coord_cols].sort_values(
                by=["target_id", "resid"]
            )

            # Create a dictionary for faster lookup: {target_id: coords_array}
            self.target_id_to_coords = {}
            if not coord_data_cols:  # No coordinate data available
                print(
                    "Warning: No coordinate columns (e.g., x_1) found in labels file."
                )
            else:
                for target_id, group in self.labels_df.groupby("target_id"):
                    # Extract coordinates, ensure they are float
                    coords = group[coord_data_cols].astype(np.float32).values
                    self.target_id_to_coords[target_id] = coords

            print(
                f"Processed labels for {len(self.target_id_to_coords)} targets in split '{self.split}'"
            )

        except Exception as e:
            print(f"Error processing labels: {e}. Check label file format.")
            import traceback

            traceback.print_exc()  # Print full traceback for debugging
            # Decide how to handle: raise error, clear labels, etc.
            self.labels_df = None  # Invalidate labels dataframe on error too
            self.target_id_to_coords = {}  # Clear processed coords

    def __len__(self):
        """Returns the number of sequences in the dataset."""
        return len(self.sequences_df)

    def __getitem__(self, idx):
        """Retrieves the idx-th sample from the dataset."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sequence_info = self.sequences_df.iloc[idx]
        target_id = sequence_info["target_id"]
        sequence = sequence_info["sequence"]

        # Sample dictionary - start with base info
        sample = {"target_id": target_id, "sequence": sequence}

        # Get labels (coordinates)
        coordinates = None
        if self.target_id_to_coords and target_id in self.target_id_to_coords:
            coords_np = self.target_id_to_coords[target_id]
            # Validate length consistency (important check)
            if len(sequence) != coords_np.shape[0]:
                print(
                    f"Warning: Sequence length {len(sequence)} mismatch with coordinates length {coords_np.shape[0]} for target {target_id}. Skipping coordinates."
                )
                # Handle mismatch: return None for coords, skip sample, error? Decide based on use case.
            else:
                coordinates = torch.tensor(coords_np, dtype=torch.float32)
                sample["coordinates"] = coordinates

        # Get MSA data
        msa_data = []
        msa_file_path = self.msa_dir / f"{target_id}.MSA.fasta"
        if msa_file_path.exists():
            msa_data = parse_fasta_msa(msa_file_path)
            # Optionally add basic validation (e.g., check if non-empty)
            # if not msa_data:
            #     print(f"Warning: Parsed empty MSA data for {target_id}")
        # else:
        # print(f"Debug: MSA file not found for {target_id} at {msa_file_path}")
        # Add MSA data to sample (even if empty list, indicates file processed/not found)
        sample["msa"] = msa_data

        # Apply transforms if any
        if self.transform:
            sample = self.transform(sample)

        return sample


# Example Usage (optional, for testing):
if __name__ == "__main__":
    # Adjust the path to where your data actually is
    # Example: python -m protein_rna_3d.datasets --data_dir ../data/stanford-rna-3d-folding

    parser = argparse.ArgumentParser(description="Test RNA3DDataset Loading")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data/stanford-rna-3d-folding",  # Default relative path
        help="Path to the root directory of the competition data.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="Data split to load.",
    )
    parser.add_argument(
        "--num_samples", type=int, default=3, help="Number of samples to print."
    )
    args = parser.parse_args()

    DATA_ROOT = Path(args.data_dir)

    if not DATA_ROOT.is_dir():
        print(
            f"Error: Provided data directory does not exist or is not a directory: {DATA_ROOT}"
        )
        print("Please provide the correct path using --data_dir")
    else:
        print(f"Attempting to load {args.split} data from: {DATA_ROOT}")
        try:
            dataset = RNA3DDataset(data_dir=DATA_ROOT, split=args.split)
            print(
                f"Successfully loaded {args.split} dataset with {len(dataset)} samples."
            )

            if len(dataset) == 0:
                print("Dataset is empty.")
            else:
                num_to_show = min(args.num_samples, len(dataset))
                print(f"\nShowing first {num_to_show} sample(s):")
                for i in range(num_to_show):
                    try:
                        sample = dataset[i]
                        print(f"\n--- Sample {i} ---")
                        print(f"  Target ID: {sample.get('target_id', 'N/A')}")
                        seq = sample.get("sequence", "")
                        print(
                            f"  Sequence (len {len(seq)}): {seq[:50]}{'...' if len(seq) > 50 else ''}"
                        )
                        if "coordinates" in sample:
                            coords = sample["coordinates"]
                            print(f"  Coordinates shape: {coords.shape}")
                            # print(f"  Coordinates dtype: {coords.dtype}")
                            # print(f"  Coordinates (first 2):\n{coords[:2].numpy()}")
                        else:
                            print("  Coordinates: Not available")
                        # Print MSA info
                        if "msa" in sample:
                            msa = sample["msa"]
                            print(f"  MSA sequences loaded: {len(msa)}")
                            if msa:
                                print(
                                    f"    MSA seq 0 (len {len(msa[0])}): {msa[0][:50]}{'...' if len(msa[0]) > 50 else ''}"
                                )
                        else:
                            print("  MSA: Key not found in sample")

                    except Exception as e:
                        print(f"Error retrieving or printing sample {i}: {e}")

        except FileNotFoundError as e:
            print(f"\nError: Could not find necessary file. {e}")
            print(f"Please ensure '{args.split}_sequences.csv' exists in {DATA_ROOT}.")
            if args.split != "test":
                print(f"Also check for '{args.split}_labels.csv' if applicable.")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            import traceback

            traceback.print_exc()
