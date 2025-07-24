# [2] Improved Transformer Model for Bitcoin Price Prediction
# Author: Sirui Zeng, Zhaoyang Pan
# Course: Data 612 Deep Learning


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

# Set device and random seeds
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)

class PositionalEncoding(nn.Module):
    """Positional Encoding for time series"""
    def __init__(self, d_model, max_len=1000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class ImprovedBitcoinTransformer(nn.Module):
    """
    Improved Transformer model with better scaling and architecture
    """
    
    def __init__(self, input_dim=11, d_model=128, nhead=4, num_layers=3, 
                 dim_feedforward=256, dropout=0.2, seq_length=60):
        super(ImprovedBitcoinTransformer, self).__init__()
        
        self.d_model = d_model
        self.seq_length = seq_length
        
        # Input projection with layer norm
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_length, dropout=dropout)
        
        # Transformer encoder (smaller and more focused)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',  # Better activation
            batch_first=False
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Output layers with residual connection
        self.output_layers = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 32),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(32, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Better weight initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, src):
        # Input shape: (batch_size, seq_len, features)
        batch_size, seq_len, features = src.shape
        
        # Project input to model dimension
        src = self.input_projection(src)
        
        # Transpose for transformer: (seq_len, batch_size, d_model)
        src = src.transpose(0, 1)
        
        # Add positional encoding
        src = self.pos_encoder(src)
        
        # Pass through transformer
        transformer_output = self.transformer_encoder(src)
        
        # Global average pooling instead of just last timestep
        pooled_output = transformer_output.mean(dim=0)  # (batch_size, d_model)
        
        # Generate prediction
        output = self.output_layers(pooled_output)
        
        return output.squeeze(-1)

class ImprovedDataProcessor:
    """
    Improved data processing with better target scaling
    """
    
    def __init__(self):
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        
    def prepare_data_improved(self, seq_length=60):
        """Load and process data with separate target scaling"""
        print("Loading and reprocessing data")
        
        # Load processed data
        df = pd.read_csv('bitcoin_processed.csv')
        df['open_time'] = pd.to_datetime(df['open_time'])
        df = df.sort_values('open_time')
        
        # Select features
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 
                       'ma_5', 'ma_20', 'ma_50', 'rsi', 'volatility', 'volume_ratio']
        
        # Scale features
        features_scaled = self.feature_scaler.fit_transform(df[feature_cols])
        
        # Scale target separately (close price)
        target_scaled = self.target_scaler.fit_transform(df[['close']])
        
        # Create sequences
        X, y = [], []
        for i in range(seq_length, len(features_scaled)):
            X.append(features_scaled[i-seq_length:i])
            y.append(target_scaled[i, 0])  # Use scaled close price
        
        X, y = np.array(X), np.array(y)
        
        print(f"Created {len(X):,} sequences")
        print(f"Feature range: [{features_scaled.min():.3f}, {features_scaled.max():.3f}]")
        print(f"Target range: [{target_scaled.min():.3f}, {target_scaled.max():.3f}]")
        
        return X, y
    
    def save_scalers(self):
        """Save both scalers"""
        joblib.dump(self.feature_scaler, 'feature_scaler.pkl')
        joblib.dump(self.target_scaler, 'target_scaler.pkl')
        print("Scalers saved")

class ImprovedTrainer:
    """Improved training with better hyperparameters"""
    
    def __init__(self, model, device=device):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        
    def train(self, train_loader, val_loader, epochs=50, lr=0.0001, patience=10):
        """Training with lower learning rate and better optimization"""
        print(f"Starting improved training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Better optimizer setup
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )
        
        # Smoother learning rate scheduling
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=lr/10
        )
        
        criterion = nn.MSELoss()
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            # Training
            self.model.train()
            train_loss = 0
            num_batches = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                
                optimizer.step()
                train_loss += loss.item()
                num_batches += 1
            
            # Validation
            self.model.eval()
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    loss = criterion(output, target)
                    val_loss += loss.item()
                    val_batches += 1
            
            train_loss /= num_batches
            val_loss /= val_batches
            
            # Update learning rate
            scheduler.step()
            
            # Record losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_improved_model.pth')
            else:
                patience_counter += 1
            
            # Print progress every 5 epochs
            if epoch % 5 == 0 or epoch == 1:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch {epoch:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | '
                      f'LR: {current_lr:.2e} | Patience: {patience_counter}/{patience}')
                
            
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_improved_model.pth'))
        print(f'Training completed. Best val loss: {best_val_loss:.6f}')
        
        return self.train_losses, self.val_losses

class ImprovedEvaluator:
    """Improved evaluation with proper scaling"""
    
    def __init__(self, model, feature_scaler, target_scaler, device=device):
        self.model = model.to(device)
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.device = device
    
    def predict(self, test_loader):
        """Generate predictions"""
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                predictions.extend(output.cpu().numpy())
                actuals.extend(target.cpu().numpy())
        
        return np.array(predictions), np.array(actuals)
    
    def evaluate_metrics(self, predictions, actuals):
        """Calculate evaluation metrics with proper inverse scaling"""
        # Convert back to original scale
        pred_original = self.target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        actual_original = self.target_scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        mae = mean_absolute_error(actual_original, pred_original)
        rmse = np.sqrt(mean_squared_error(actual_original, pred_original))
        mape = np.mean(np.abs((actual_original - pred_original) / actual_original)) * 100
        r2 = r2_score(actual_original, pred_original)
        
        # Directional accuracy
        actual_direction = np.diff(actual_original) > 0
        pred_direction = np.diff(pred_original) > 0
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2,
            'Directional_Accuracy': directional_accuracy,
            'predictions_original': pred_original,
            'actuals_original': actual_original
        }

def create_improved_data_loaders(X, y, batch_size=32, train_ratio=0.8):
    """Create data loaders with proper splitting"""
    # Time-based split (important for time series)
    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)
    
    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    return train_loader, test_loader

def plot_improved_results(predictions, actuals, train_losses, val_losses):
    """Plot improved results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training history
    axes[0,0].plot(train_losses, label='Train Loss', alpha=0.8)
    axes[0,0].plot(val_losses, label='Validation Loss', alpha=0.8)
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].set_title('Training History')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Predictions vs actual (time series)
    show_points = min(1000, len(predictions))
    axes[0,1].plot(actuals[-show_points:], label='Actual', alpha=0.8, linewidth=1.5)
    axes[0,1].plot(predictions[-show_points:], label='Predicted', alpha=0.8, linewidth=1.5)
    axes[0,1].set_xlabel('Time Steps')
    axes[0,1].set_ylabel('Price (USDT)')
    axes[0,1].set_title(f'Price Predictions (Last {show_points} points)')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Scatter plot
    axes[1,0].scatter(actuals, predictions, alpha=0.5, s=1)
    min_val = min(actuals.min(), predictions.min())
    max_val = max(actuals.max(), predictions.max())
    axes[1,0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    axes[1,0].set_xlabel('Actual Price')
    axes[1,0].set_ylabel('Predicted Price')
    axes[1,0].set_title('Prediction vs Actual')
    axes[1,0].grid(True, alpha=0.3)
    
    # Error distribution
    errors = predictions - actuals
    axes[1,1].hist(errors, bins=50, alpha=0.7, density=True)
    axes[1,1].axvline(errors.mean(), color='red', linestyle='--', 
                     label=f'Mean: ${errors.mean():.2f}')
    axes[1,1].set_xlabel('Prediction Error (USDT)')
    axes[1,1].set_ylabel('Density')
    axes[1,1].set_title('Error Distribution')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('improved_model_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function with improved pipeline"""
    print("Improved Bitcoin Transformer Model")
    print("=" * 60)
    
    # Step 1: Improved data processing
    processor = ImprovedDataProcessor()
    X, y = processor.prepare_data_improved(seq_length=60)
    processor.save_scalers()
    
    # Step 2: Create data loaders
    train_loader, test_loader = create_improved_data_loaders(X, y, batch_size=64)
    
    # Step 3: Create improved model (smaller, more focused)
    model = ImprovedBitcoinTransformer(
        input_dim=11,
        d_model=128,      # Smaller model
        nhead=4,          # Fewer heads
        num_layers=3,     # Fewer layers
        dim_feedforward=256,
        dropout=0.2,
        seq_length=60
    )
    
    print(f"Improved Model:")
    print(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   - Model size: {sum(p.numel() * 4 for p in model.parameters()) / 1024**2:.2f} MB")
    
    # Step 4: Train model
    trainer = ImprovedTrainer(model, device)
    train_losses, val_losses = trainer.train(
        train_loader, test_loader,
        epochs=100,
        lr=0.0001,  # Lower learning rate
        patience=15
        
    )
    
    # Step 5: Evaluate
    feature_scaler = processor.feature_scaler
    target_scaler = processor.target_scaler
    
    evaluator = ImprovedEvaluator(model, feature_scaler, target_scaler, device)

    # running time of prediction
    print("running time of prediction")
    start_time = time.time()
    predictions, actuals = evaluator.predict(test_loader)
    inference_time = time.time() - start_time
    print(f"Inference time on test set: {inference_time:.4f} seconds")

    # save as txt
    with open("inference_time.txt", "w") as f:
        f.write(f"Inference time on test set: {inference_time:.4f} seconds\n")


    metrics = evaluator.evaluate_metrics(predictions, actuals)
    
    # Step 6: Results
    plot_improved_results(
        metrics['predictions_original'], 
        metrics['actuals_original'],
        train_losses, 
        val_losses
    )
    
    # Final report
    print("\n" + "=" * 60)
    print("IMPROVED MODEL RESULTS")
    print("=" * 60)
    
    print(f"Performance Metrics:")
    print(f"   - MAE: ${metrics['MAE']:.2f}")
    print(f"   - RMSE: ${metrics['RMSE']:.2f}")
    print(f"   - MAPE: {metrics['MAPE']:.2f}%")
    print(f"   - R² Score: {metrics['R2']:.4f}")
    print(f"   - Directional Accuracy: {metrics['Directional_Accuracy']:.2f}%")
    
    # Grade assessment
    if metrics['R2'] > 0.7 and metrics['MAPE'] < 10:
        grade = "23-25/25 (Excellent)"
    elif metrics['R2'] > 0.5 and metrics['MAPE'] < 15:
        grade = "20-22/25 (Very Good)"
    elif metrics['R2'] > 0.3 and metrics['MAPE'] < 25:
        grade = "17-19/25 (Good)"
    else:
        grade = "15-16/25 (Needs More Improvement)"
    
    print(f"\nExpected Grade: {grade}")
    
    return model, metrics

if __name__ == "__main__":
    model, results = main()
