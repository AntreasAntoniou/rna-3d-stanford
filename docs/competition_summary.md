# Stanford RNA 3D Folding Competition Summary

## Overview

This Kaggle competition challenges participants to develop machine learning models capable of predicting the 3D structure of RNA molecules based solely on their nucleotide sequences. The goal aligns with advancing our understanding of fundamental biological processes and driving innovation in RNA-based medicine, including therapies and gene editing technologies.

## Key Challenge

Predicting RNA 3D structures is significantly harder than protein structure prediction (e.g., AlphaFold) due to limited data and evaluation methods. This competition aims to build upon recent advancements like the RibonanzaNet foundation model from a previous Kaggle challenge.

## Evaluation

*   **Metric:** TM-score (0.0 to 1.0, higher is better), comparing predicted structures to experimental reference structures.
*   **Alignment:** US-align performs sequence-independent alignment.
*   **Submission:** For each target sequence, submit 5 predicted structures.
*   **Final Score:** Average of the best TM-score (out of 5 predictions) for each target sequence.

## Submission Details

*   **Input:** `test_sequences.csv`
*   **Output:** `submission.csv` containing x, y, z coordinates of the C1' atom for each residue across the 5 predictions.
*   **Format:** See `README.md` for the exact CSV format.

## Timeline

*   **Start:** Feb 27, 2025
*   **Entry Deadline:** May 22, 2025
*   **Submission Deadline:** May 29, 2025
*   **End (Tentative):** Sep 24, 2025 (with future data evaluation)

## Code Requirements

*   Kaggle Notebooks.
*   Runtime limits: 8 hours (CPU/GPU).
*   No internet access during submission run.
*   External data/models allowed if publicly available.

## Collaboration & Prizes

The competition involves collaboration with CASP16, RNA-Puzzles, HHMI, IPD, and Stanford. Prizes are available for top leaderboard positions and early sharing of high-performing public notebooks. 