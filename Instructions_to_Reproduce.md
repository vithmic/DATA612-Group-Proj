# Instructions to Reproduce Results

## Prerequisites
- Python 3.8+ (tested on Python 3.10)
- Windows/macOS/Linux

## Setup
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download dataset:**
   - Download `btc_1h_data_2018_to_2025.csv` from [Kaggle Bitcoin Dataset](https://www.kaggle.com/datasets/novandraanugrah/bitcoin-historical-datasets-2018-2024)
   - Place the file in the project root directory

## Reproduction Steps

### Step 1: Data Preprocessing
```bash
python clean_Step1.py
```
**Output files:** `bitcoin_processed.csv`, `bitcoin_sequences_X.npy`, `bitcoin_sequences_y.npy`, `bitcoin_scaler.pkl`

### Step 2: Train Baseline Model
```bash
python baseline_step2.py
```
**Expected results:** R² ≈ -3.7, MAPE ≈ 47%

### Step 3: Train Improved Transformer
```bash
python Transformer_Step2_3_4.py
```
**Expected results:** R² ≈ 0.94, MAPE ≈ 5.1%

## Verification
After running all scripts, you should have:
- `improved_model_results.png` (visualization plots)
- `best_improved_model.pth` (trained model)
- `feature_scaler.pkl`, `target_scaler.pkl` (data scalers)
- `inference_time.txt` (timing information)

## Troubleshooting
- **Memory error:** Reduce batch_size from 64 to 32 in the code
- **CUDA unavailable:** Code automatically falls back to CPU
- **File not found:** Ensure Bitcoin dataset is in project root folder

**Total runtime:** ~5-15 minutes depending on hardware
