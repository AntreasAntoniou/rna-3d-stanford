# RNA3D Stanford Competition

This repository contains code for the [Stanford RNA 3D Folding Kaggle competition](https://kaggle.com/competitions/stanford-rna-3d-folding), focused on predicting RNA 3D structures from their sequences using machine learning.

## Competition Goal

Develop machine learning models to predict an RNA molecule's 3D structure solely from its sequence. The aim is to improve the understanding of biological processes and advance RNA-based medicine.

## Evaluation Metric

Submissions are scored using the **TM-score** (Template Modeling score), ranging from 0.0 to 1.0 (higher is better). The final score is the average of the best-of-5 TM-scores across all target sequences.

TM-score formula:
\[ TM_{score} = \max \left( \frac{1}{L_{ref}} \sum_{i=1}^{L_{align}} \frac{1}{1 + (d_i/d_0)^2} \right) \]
where \(L_{ref}\) is the reference structure length, \(L_{align}\) is the number of aligned residues, \(d_i\) is the distance between the i-th pair of aligned residues, and \(d_0\) is a length-dependent scaling factor. Alignment is sequence-independent using US-align.

## Submission Format

For each sequence in `test_sequences.csv`, predict 5 structures. The output must be a `submission.csv` file containing the x, y, z coordinates of the C1' atom for each residue across the 5 predicted structures.

Format:
```csv
ID,resname,resid,x_1,y_1,z_1,... x_5,y_5,z_5
R1107_1,G,1,-7.561,9.392,9.361,... -7.301,9.023,8.932
R1107_2,G,1,-8.02,11.014,14.606,... -7.953,10.02,12.127
...
```

## Key Dates

*   **Start Date:** February 27, 2025
*   **Entry Deadline:** May 22, 2025
*   **Final Submission Deadline:** May 29, 2025
*   **Competition End Date (Subject to Change):** September 24, 2025

## Code Requirements

*   Submissions via Kaggle Notebooks.
*   CPU Notebook Runtime <= 8 hours.
*   GPU Notebook Runtime <= 8 hours.
*   Internet access must be disabled during submission run.
*   Freely & publicly available external data and pre-trained models (like RibonanzaNet) are allowed.
*   Submission file must be named `submission.csv`.

## Resources

*   [Competition Page](https://kaggle.com/competitions/stanford-rna-3d-folding)
*   [RibonanzaNet Foundation Model](https://github.com/stanford-ribonanza-project/RibonanzaNet)
*   [CASP16 Challenge](https://predictioncenter.org/casp16/)
*   [RNA-Puzzles Results](http://rnapuzzles.org/)

## Project Structure

*(To be filled in as the project develops)*

## Setup

*(Instructions for setting up the environment and dependencies)*

## Usage

*(How to run the code for training and prediction)*

## Contributing

*(Guidelines for contributing to the project)*

## License

*(Project license information)* 