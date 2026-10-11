# Similarity-Aware Transfer Learning for Battery SOH Estimation

Research implementation of similarity-aware transfer learning for
lithium-ion battery State-of-Health (SOH) estimation.

## Overview

This project investigates how battery similarity can support
the transfer of pretrained deep learning models across datasets.

The framework combines:

- LSTM-based SOH modelling
- Gradient-based charging-feature importance
- Weighted similarity assessment
- MIT-to-NASA transfer learning
- Head-only fine-tuning
- SOH reconstruction and evaluation

## Methodology

1. Train an LSTM on source battery charging sequences.
2. Calculate charging-feature importance using input gradients.
3. Estimate source-to-target similarity using a weighted distance.
4. Evaluate transfer learning using pretrained LSTM representations.
5. Reconstruct absolute SOH from predicted SOH changes.

The weighted similarity distance is:

D_w = sum_j w_j * ((x_j - mu_j) / sigma_j)^2

where w_j represents the normalised feature importance,
and mu_j and sigma_j represent source-dataset statistics.

## Model

- Architecture: 2-layer LSTM
- Hidden dimension: 64
- Dropout: 0.2
- Default sequence length: 20
- Input features: 10 charging health indicators

## Repository Structure

- src/models/ — LSTM architecture
- src/train/ — training, feature weighting, similarity and evaluation
- results/ — selected experimental outputs
- requirements.txt — Python dependencies

## Data

The research uses battery datasets including MIT and NASA.
Original datasets are not distributed in this repository.

Some scripts retain HPC-specific paths and require configuration
for another environment. This initial release is an archival
research implementation, not yet a fully reproducible package.

## Publication

Selective Model Transfer for Li-ion SOH Estimation Based on a
Weighted Similarity Metric.

Journal of Energy Storage (2026).

## Author

Asaad Challab

PhD Candidate, University of Technology Sydney

[GitHub](https://github.com/Asaad-challab) |
[LinkedIn](https://www.linkedin.com/in/asaad-wahaj-217253246/)

## Experimental Results

### MIT Battery Dataset — Anchored-Window SOH Reconstruction

| Metric | Result |
|---|---:|
| Mean MAE | 0.1137 percentage points |
| Mean RMSE | 0.1850 percentage points |
| Mean R² | 0.9977 |
| Test cells | 8 |

These results use the observed SOH at the beginning of each
prediction window to reconstruct absolute SOH from predicted
SOH changes. They are not fully autonomous rolling predictions.

### Gradient-Based Feature Importance

| Feature | Weight |
|---|---:|
| Constant-voltage charging duration (t_CV) | 0.2550 |
| Charging capacity (Q_chg) | 0.1531 |
| Charging energy (E_chg) | 0.1192 |
| Time to 60% voltage threshold (t_V60) | 0.1094 |

The complete feature weights are provided in
`results/mit_feature_weights.json`.

### Reproducibility Notes

The saved feature weights reference an MIT checkpoint trained
in December 2025. The feature-weighting script computes
absolute-SOH loss gradients, while the reviewed MIT baseline
script trains on SOH changes. This difference requires
verification before claiming exact reproduction of the
published feature-weight estimation procedure.

The original dataset files and trained checkpoints are
not included in this repository.
