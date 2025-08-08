# baseline model for Bitcoin Price Prediction
# Author: Zhaoyang Pan
# Course: Data 612 Deep Learning


import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# seed
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Baseline Transformer
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x[:, -1, :]         # last time step
        out = self.fc_out(x)
        return out.squeeze(-1)


# Load
X = np.load('bitcoin_sequences_X.npy')
y = np.load('bitcoin_sequences_y.npy')


# Scale features/target & save scalers
feature_scaler = MinMaxScaler()
target_scaler  = MinMaxScaler()

if X.ndim != 3:
    raise ValueError("Expected X with shape [N, seq_len, feat].")

N, S, F = X.shape
X_flat = X.reshape(N, S * F)                           
X_scaled_flat = feature_scaler.fit_transform(X_flat)
X_scaled = X_scaled_flat.reshape(N, S, F)
y_scaled = target_scaler.fit_transform(y.reshape(-1, 1)).ravel()

# Save scalers
joblib.dump(feature_scaler, 'feature_scaler.pkl')
joblib.dump(target_scaler,  'target_scaler.pkl')

# Save a processed CSV
cols = [f'F{j}_t{-i}' for i in range(S-1, -1, -1) for j in range(F)]
df_proc = pd.DataFrame(X_scaled.reshape(N, S * F), columns=cols)
df_proc['Target'] = y_scaled
df_proc.to_csv('bitcoin_processed.csv', index=False)


# Time-based split (80/20)
split_idx = int(N * 0.8)
idx_train = list(range(0, split_idx))
idx_val   = list(range(split_idx, N))

train_ds = Subset(TensorDataset(torch.FloatTensor(X_scaled), torch.FloatTensor(y_scaled)), idx_train)
val_ds   = Subset(TensorDataset(torch.FloatTensor(X_scaled), torch.FloatTensor(y_scaled)), idx_val)

batch_size = 64
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, drop_last=False)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)


# Init model & optimizer
model = TimeSeriesTransformer(input_dim=F).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Print model info
print("Improved Model:")
print(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"   - Model size: {sum(p.numel() * 4 for p in model.parameters()) / 1024**2:.2f} MB")


# Train
epochs = 20
train_losses, val_losses = [], []

for epoch in range(1, epochs + 1):
    model.train()
    total = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    train_loss = total / max(1, len(train_loader))

    model.eval()
    vtotal = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            vtotal += loss.item()
    val_loss = vtotal / max(1, len(val_loader))

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # Print progress (every 5 like improved script)
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")


# Evaluate & save files
# Save model weights
torch.save(model.state_dict(), 'best_improved_model.pth')

# Inference timing on the test set
t0 = time.time()
y_true_list, y_pred_list = [], []
model.eval()
with torch.no_grad():
    for xb, yb in val_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        y_true_list.append(yb.detach().cpu().view(-1))
        y_pred_list.append(pred.detach().cpu().view(-1))
infer_time = time.time() - t0

with open("inference_time.txt", "w") as f:
    f.write(f"Inference time on test set: {infer_time:.4f} seconds\n")

y_true = torch.cat(y_true_list).numpy()
y_pred = torch.cat(y_pred_list).numpy()

# Inverse scale to original
y_true_orig = target_scaler.inverse_transform(y_true.reshape(-1, 1)).ravel()
y_pred_orig = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()

# Metrics
mae  = mean_absolute_error(y_true_orig, y_pred_orig)
rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
mape = np.mean(np.abs((y_true_orig - y_pred_orig) / (y_true_orig + 1e-8))) * 100
r2   = r2_score(y_true_orig, y_pred_orig)
actual_direction = np.diff(y_true_orig) > 0
pred_direction   = np.diff(y_pred_orig) > 0
directional_acc  = np.mean(actual_direction == pred_direction) * 100

metrics = {
    'MAE': mae,
    'RMSE': rmse,
    'MAPE': mape,
    'R2': r2,
    'Directional_Accuracy': directional_acc
}


# Plot 2x2 figure
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# (0,0) Training history
axes[0,0].plot(train_losses, label='Train Loss', alpha=0.8)
axes[0,0].plot(val_losses,   label='Validation Loss', alpha=0.8)
axes[0,0].set_xlabel('Epoch')
axes[0,0].set_ylabel('Loss')
axes[0,0].set_title('Training History')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# (0,1) Predictions vs Actual
show_points = min(1000, len(y_pred_orig))
axes[0,1].plot(y_true_orig[-show_points:], label='Actual', alpha=0.8, linewidth=1.5)
axes[0,1].plot(y_pred_orig[-show_points:], label='Predicted', alpha=0.8, linewidth=1.5)
axes[0,1].set_xlabel('Time Steps')
axes[0,1].set_ylabel('Price (USDT)')
axes[0,1].set_title(f'Price Predictions (Last {show_points} points)')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# (1,0) Scatter y_true vs y_pred with y=x
axes[1,0].scatter(y_true_orig, y_pred_orig, alpha=0.5, s=1)
min_val = min(y_true_orig.min(), y_pred_orig.min())
max_val = max(y_true_orig.max(), y_pred_orig.max())
axes[1,0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
axes[1,0].set_xlabel('Actual Price')
axes[1,0].set_ylabel('Predicted Price')
axes[1,0].set_title('Prediction vs Actual')
axes[1,0].grid(True, alpha=0.3)

# (1,1) Error distribution
errors = y_pred_orig - y_true_orig
axes[1,1].hist(errors, bins=50, alpha=0.7, density=True)
axes[1,1].axvline(errors.mean(), color='red', linestyle='--', label=f'Mean: ${errors.mean():.2f}')
axes[1,1].set_xlabel('Prediction Error (USDT)')
axes[1,1].set_ylabel('Density')
axes[1,1].set_title('Error Distribution')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('improved_model_results.png', dpi=300, bbox_inches='tight')
plt.show()


# Final report
print("IMPROVED MODEL RESULTS")
print(f"Performance Metrics:")
print(f"   - MAE: {metrics['MAE']:.2f}")
print(f"   - RMSE: {metrics['RMSE']:.2f}")
print(f"   - MAPE: {metrics['MAPE']:.2f}%")
print(f"   - R^2 Score: {metrics['R2']:.4f}")
print(f"   - Directional Accuracy: {metrics['Directional_Accuracy']:.2f}%")

if metrics['R2'] > 0.7 and metrics['MAPE'] < 10:
    grade = "23-25/25 (Excellent)"
elif metrics['R2'] > 0.5 and metrics['MAPE'] < 15:
    grade = "20-22/25 (Very Good)"
elif metrics['R2'] > 0.3 and metrics['MAPE'] < 25:
    grade = "17-19/25 (Good)"
else:
    grade = "15-16/25 (Needs More Improvement)"

print(f"\nExpected Grade: {grade}")
print("Saved: improved_model_results.png, best_improved_model.pth, inference_time.txt, feature_scaler.pkl, target_scaler.pkl, bitcoin_processed.csv")
