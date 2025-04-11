import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import argparse


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
            self.labels_df[["target_id", "resid_str"]] = self.labels_df[
                "ID"
            ].str.rsplit("_", n=1, expand=True)
            self.labels_df["resid"] = self.labels_df["resid"].astype(int)
            # Keep only essential columns for coordinate extraction
            # For training, we primarily use the first structure (_1)
            coord_cols = ["target_id", "resid", "x_1", "y_1", "z_1"]
            # Check if columns exist, handle cases with multiple structures if needed later
            available_coord_cols = [
                c for c in coord_cols if c in self.labels_df.columns
            ]
            if (
                len(available_coord_cols) < 5 and self.split == "train"
            ):  # Expect x_1,y_1,z_1 for train
                print(
                    f"Warning: Missing expected coordinate columns (x_1, y_1, z_1) in {self.split}_labels.csv"
                )
                # Decide how to handle this: error out, return None, etc.
                # For now, let it proceed but downstream code will fail if coords are needed

            # Pivot and group by target_id to get coordinates per sequence
            # Sort by resid to ensure correct order
            self.labels_df = self.labels_df[available_coord_cols].sort_values(
                by=["target_id", "resid"]
            )
            # Create a dictionary for faster lookup: {target_id: coords_array}
            self.target_id_to_coords = {}
            coord_data_cols = [
                c for c in available_coord_cols if c not in ["target_id", "resid"]
            ]

            for target_id, group in self.labels_df.groupby("target_id"):
                # Extract coordinates, ensure they are float
                coords = group[coord_data_cols].astype(np.float32).values
                self.target_id_to_coords[target_id] = coords

            print(
                f"Processed labels for {len(self.target_id_to_coords)} targets in split '{self.split}'"
            )

        except Exception as e:
            print(f"Error processing labels: {e}. Check label file format.")
            # Decide how to handle: raise error, clear labels, etc.
            self.labels_df = None  # Invalidate labels on error
            self.target_id_to_coords = {}

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

        # Get labels (coordinates)
        coordinates = None
        if self.labels_df is not None and target_id in self.target_id_to_coords:
            coords_np = self.target_id_to_coords[target_id]
            # Validate length consistency (important check)
            if len(sequence) != coords_np.shape[0]:
                print(
                    f"Warning: Sequence length {len(sequence)} mismatch with coordinates length {coords_np.shape[0]} for target {target_id}. Skipping coordinates."
                )
                # Handle mismatch: return None for coords, skip sample, error? Decide based on use case.
            else:
                coordinates = torch.tensor(coords_np, dtype=torch.float32)

        # Sample dictionary
        # Input: sequence ; Output: coordinates (if available)
        sample = {"target_id": target_id, "sequence": sequence}
        if coordinates is not None:
            sample["coordinates"] = coordinates

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
                            print(f"  Coordinates dtype: {coords.dtype}")
                            print(
                                f"  Coordinates (first 2):\n{coords[:2].numpy()}"
                            )  # Print as numpy for readability
                        else:
                            print("  Coordinates: Not available")
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
