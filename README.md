# Transformer Model for Bitcoin Price Prediction

## Project Overview
This project explores the use of a Transformer-based neural network for hourly Bitcoin price forecasting. Unlike traditional models (ARIMA, LSTM), our approach leverages multi-head self-attention, positional encoding, residual connections, and financial-aware features to capture long-term dependencies and non-linear dynamics in cryptocurrency markets.

Course: DATA 612 – Deep Learning

Team Members: Sirui Zeng, Zhaoyang Pan, Yunlong Ou, Yuyun Zhen, Ruikang Yan

## Dataset

Source: BITCOIN Historical Datasets 2018-2025 Binance API

Link: https://www.kaggle.com/datasets/novandraanugrah/bitcoin-historical-datasets-2018-2024

Preprocessing: Gap detection, normalization, anomaly removal, sequence generation

## Model Architecture

Input Dimension: 11 features × 60 timesteps

Model Dimension (d_model): 128

Attention Heads: 4

Encoder Layers: 3

Feedforward Dimension: 256

Dropout Rate: 0.2

## Results

| Metric                | Baseline Model           | Improved Model        |
|------------------------|--------------------------|-----------------------|
| MAE (Mean Abs Error)   | $0.34                   | $3874.63              |
| RMSE                   | $0.38                   | $4649.35              |
| MAPE                   | 47.53 %                 | 5.13 %                |
| R² Score               | –3.7289                 | 0.9414                |
| Directional Accuracy   | 49.30 %                 | 49.53 %               |
| Inference Time         | 0.66 seconds            | 0.6642 seconds        |
| Parameters             | 563,137 (~2.15 MB)      | ~260,000 (~1.0 MB)    |

## Tools & Library

Python 3.11 · PyTorch 2.1 · CUDA 12.4 · NumPy · Pandas · scikit‑learn · Matplotlib · Binance API · Kaggle CLI · Weights & Biases · GitHub Actions

## Future Work

Improve directional accuracy with classification ensembles

Add on-chain and order-book features

Explore multi-resolution inputs (minute/hour/day)

Develop probabilistic Transformers for predictive intervals
