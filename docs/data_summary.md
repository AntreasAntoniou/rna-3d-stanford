# RNA 3D Folding Competition Data Summary

This document summarizes the data provided for the Stanford RNA 3D Folding Kaggle competition.

## Overview

The goal is to predict five 3D structures (specifically, the coordinates of the C1' atom for each residue) for each given RNA sequence.

The competition proceeds in phases, with the test set being updated, potentially requiring model retraining or adaptation.

## File Structure

The primary data files are:

*   **`[train/validation/test]_sequences.csv`**: Contains the RNA sequences and metadata.
    *   `target_id`: Unique identifier for the RNA molecule. Format in `train_sequences.csv` is often `pdb_id_chain_id`.
    *   `sequence`: The RNA sequence string (ACGU). Train sequences might contain other characters.
    *   `temporal_cutoff`: Publication date (yyyy-mm-dd). Crucial for creating valid training/validation splits, especially for early sharing prizes.
    *   `description`: Details about the sequence origin, potentially including ligands (ligand coordinates are not predicted).
    *   `all_sequences`: FASTA-formatted sequences of all molecules in the solved structure (can include partners like proteins, DNA, other RNAs).

*   **`[train/validation]_labels.csv`**: Contains the experimental structural information (ground truth).
    *   `ID`: Identifier linking to `target_id` and residue number (`target_id_resid`). Residue numbers are 1-based.
    *   `resname`: The RNA nucleotide character (A, C, G, U).
    *   `resid`: Residue number (1-based).
    *   `x_n, y_n, z_n`: Coordinates (Angstroms) of the C1' atom for the *n*-th experimentally determined conformation. `train_labels.csv` typically has one structure (`x_1, y_1, z_1`), while `validation_labels.csv` may have multiple.

*   **`sample_submission.csv`**: Shows the required format for submission.
    *   Same columns as `labels.csv` but requires coordinates for 5 predicted structures (`x_1, y_1, z_1, ..., x_5, y_5, z_5`).

*   **`MSA/` directory**: Contains Multiple Sequence Alignments in FASTA format (`{target_id}.MSA.fasta`) for each target. These are accessible to the notebook during evaluation on the hidden test set.

## Key Considerations

*   **Multiple Conformations**: Some targets have multiple experimental structures provided in the labels file. The evaluation uses the best TM-score against *any* of the provided reference structures for a given target.
*   **Temporal Splitting**: Pay attention to `temporal_cutoff` when creating validation sets to avoid data leakage, especially before the first leaderboard refresh (use data before 2022-05-27 for validation against the initial public `validation_sequences.csv`).
*   **Sequence Deduplication**: `train_sequences.csv` may contain entries for the same sequence from different PDB entries or chains. Consider deduplicating and potentially merging label information.
*   **External Data**: Pre-trained models (like RibonanzaNet) and additional datasets (like the RFdiffusion synthetic dataset) can be used, but temporal validity must be considered, especially for models trained on PDB data released after the test set's `temporal_cutoff`.
*   **Input/Output for Modeling**: The primary task is mapping `sequence` (input) to C1' coordinates (output). 