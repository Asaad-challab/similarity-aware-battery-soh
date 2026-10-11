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
