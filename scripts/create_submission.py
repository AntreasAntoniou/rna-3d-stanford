import argparse
import json
import sys

import numpy as np
import pandas as pd


def load_predictions(predictions_path: str) -> dict[str, list[np.ndarray]]:
    """Loads predictions from a JSON file.

    Expects a JSON object (dictionary) where keys are sequence IDs and values are lists
    of 5 lists of lists, each inner list representing [x, y, z] coordinates.
    Example element: "seq_id": [[[x1,y1,z1], [x2,y2,z2], ...], [[...]], ..., [[...]]]
    """
    try:
        with open(predictions_path, "r") as f:
            predictions_raw = json.load(f)

        # --- Validation and Conversion ---
        if not isinstance(predictions_raw, dict):
            raise TypeError("Predictions JSON should contain an object (dictionary).")

        predictions_processed: dict[str, list[np.ndarray]] = {}
        for seq_id, preds_list_raw in predictions_raw.items():
            if not isinstance(preds_list_raw, list) or len(preds_list_raw) != 5:
                raise ValueError(
                    f"Predictions for sequence {seq_id} should be a list of 5 arrays/lists."
                )

            processed_preds_for_seq = []
            for pred_raw in preds_list_raw:
                # Attempt conversion to numpy array early for validation
                try:
                    pred_np = np.array(pred_raw, dtype=float)
                except ValueError as e:
                    raise TypeError(
                        f"Coordinates for sequence {seq_id} must be numeric: {e}"
                    )

                if pred_np.ndim != 2 or pred_np.shape[1] != 3:
                    raise ValueError(
                        f"Prediction array/list for sequence {seq_id} must have shape (L, 3). Found {pred_np.shape}"
                    )
                processed_preds_for_seq.append(pred_np)

            # Check all 5 predictions for a sequence have the same length (L)
            lengths = {p.shape[0] for p in processed_preds_for_seq}
            if len(lengths) > 1:
                raise ValueError(
                    f"Prediction arrays for sequence {seq_id} have inconsistent lengths: {lengths}."
                )
            predictions_processed[seq_id] = processed_preds_for_seq
        # --- End Validation ---

        print(
            f"Successfully loaded and validated predictions for {len(predictions_processed)} sequences from {predictions_path}"
        )
        return predictions_processed

    except FileNotFoundError:
        print(
            f"Error: Predictions file not found at {predictions_path}", file=sys.stderr
        )
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON predictions file: {e}", file=sys.stderr)
        sys.exit(1)
    except (TypeError, ValueError) as e:
        print(f"Error validating predictions data: {e}", file=sys.stderr)
        sys.exit(1)


def create_submission(
    predictions: dict[str, list[np.ndarray]],
    sequences_df: pd.DataFrame,
    output_path: str,
):
    """Generates the submission CSV file."""
    submission_data = []
    required_columns = ["ID", "resname", "resid"] + [
        f"{coord}_{i}" for i in range(1, 6) for coord in ["x", "y", "z"]
    ]

    print(f"Processing {len(sequences_df)} sequences...")

    for _, row in sequences_df.iterrows():
        seq_id = row["sequence_id"]
        sequence = row["sequence"]
        num_residues = len(sequence)

        if seq_id not in predictions:
            print(
                f"Warning: No predictions found for sequence {seq_id}. Skipping.",
                file=sys.stderr,
            )
            continue

        preds_list = predictions[seq_id]

        # Validate prediction length against sequence length
        pred_len = preds_list[0].shape[0]
        if pred_len != num_residues:
            print(
                f"Warning: Prediction length ({pred_len}) does not match sequence "
                f"length ({num_residues}) for sequence {seq_id}. Skipping.",
                file=sys.stderr,
            )
            continue

        for i, residue_char in enumerate(sequence):
            resid = i + 1  # Residue IDs are 1-based
            resname = residue_char
            entry_id = f"{seq_id}_{resid}"

            row_data = {"ID": entry_id, "resname": resname, "resid": resid}

            for pred_idx, coords_array in enumerate(preds_list):
                coords = coords_array[i]  # Get coords for the i-th residue
                model_num = pred_idx + 1
                row_data[f"x_{model_num}"] = coords[0]
                row_data[f"y_{model_num}"] = coords[1]
                row_data[f"z_{model_num}"] = coords[2]

            submission_data.append(row_data)

    if not submission_data:
        print(
            "Error: No valid submission data generated. Check predictions and sequence file.",
            file=sys.stderr,
        )
        sys.exit(1)

    submission_df = pd.DataFrame(submission_data)

    # Ensure columns are in the correct order
    submission_df = submission_df[required_columns]

    try:
        submission_df.to_csv(output_path, index=False)
        print(
            f"Successfully created submission file at {output_path} with {len(submission_df)} rows."
        )
    except IOError as e:
        print(f"Error writing submission file: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Kaggle submission file for RNA 3D folding competition."
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to the JSON file containing predictions dictionary.",
    )
    parser.add_argument(
        "--sequences",
        required=True,
        help="Path to the CSV file containing sequence information (e.g., test_sequences.csv). Needs 'sequence_id' and 'sequence' columns.",
    )
    parser.add_argument(
        "--output",
        default="submission.csv",
        help="Path for the output submission CSV file (default: submission.csv).",
    )

    args = parser.parse_args()

    # Load sequence info
    try:
        sequences_df = pd.read_csv(args.sequences)
        if (
            "sequence_id" not in sequences_df.columns
            or "sequence" not in sequences_df.columns
        ):
            raise ValueError(
                "Sequence file must contain 'sequence_id' and 'sequence' columns."
            )
        print(f"Loaded sequence information from {args.sequences}")
    except FileNotFoundError:
        print(f"Error: Sequence file not found at {args.sequences}", file=sys.stderr)
        sys.exit(1)
    except (pd.errors.ParserError, ValueError) as e:
        print(f"Error reading or validating sequence file: {e}", file=sys.stderr)
        sys.exit(1)

    # Load predictions
    predictions = load_predictions(args.predictions)

    # Create submission file
    create_submission(predictions, sequences_df, args.output)


if __name__ == "__main__":
    main()
