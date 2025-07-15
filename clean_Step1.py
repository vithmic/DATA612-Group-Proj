# [1] Data Preparation for Bitcoin Price Prediction
# Course: Data 612 Deep Learning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

class BitcoinDataProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.processed_data = None
    
    def check_time_gaps(self, df):
        """Check for missing time intervals in the data"""
        print("Checking time continuity...")
        
        df_sorted = df.sort_values('open_time')
        deltas = df_sorted['open_time'].diff().dropna()
        expected_interval = pd.Timedelta(hours=1)
        
        # Count gaps (non-1h intervals)
        gap_mask = deltas != expected_interval
        gap_count = gap_mask.sum()
        
        print(f"Expected interval: {expected_interval}")
        print(f"Detected {gap_count} time gaps (non-1h intervals)")
        
        if gap_count > 0:
            gaps = deltas[gap_mask]
            print(f"   - Largest gap: {gaps.max()}")
            print(f"   - Average gap: {gaps.mean()}")
            print(f"   - Gap frequency: {gap_count/len(df)*100:.2f}% of records")
            
            # Show some examples
            gap_indices = gaps.head(3).index
            print("   - Example gaps:")
            for idx in gap_indices:
                prev_time = df_sorted.loc[idx-1, 'open_time']
                curr_time = df_sorted.loc[idx, 'open_time']
                print(f"     {prev_time} -> {curr_time} (gap: {curr_time - prev_time})")
        else:
            print("No time gaps detected - data is continuous")
        
        return gap_count
    
    def save_scaler(self, path='bitcoin_scaler.pkl'):
        """Save the fitted scaler for later use"""
        joblib.dump(self.scaler, path)
        print(f"Scaler saved to {path}")
    
    def load_scaler(self, path='bitcoin_scaler.pkl'):
        """Load a previously fitted scaler"""
        try:
            self.scaler = joblib.load(path)
            print(f"Scaler loaded from {path}")
            return True
        except FileNotFoundError:
            print(f"Scaler file {path} not found")
            return False
        
    def load_and_clean_data(self, file_path):
        """Load and clean Bitcoin data"""
        print("Loading Bitcoin data...")
        
        # Load data
        df = pd.read_csv(file_path)
        
        # Rename columns to standard format
        column_mapping = {
            'Open time': 'open_time',
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        df = df.rename(columns=column_mapping)
        
        # Convert time and sort
        df['open_time'] = pd.to_datetime(df['open_time'])
        df = df.sort_values('open_time').reset_index(drop=True)
        
        # Remove invalid data
        df = df[(df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)]
        
        print(f"Loaded {len(df):,} records from {df['open_time'].min()} to {df['open_time'].max()}")
        return df
    
    def create_features(self, df):
        """Create technical indicators and features"""
        print("Creating features...")
        
        # Basic features
        df['price_change'] = df['close'].pct_change()
        df['volatility'] = df['close'].rolling(20).std()
        
        # Moving averages
        for period in [5, 20, 50]:
            df[f'ma_{period}'] = df['close'].rolling(period).mean()
            df[f'price_to_ma_{period}'] = df['close'] / df[f'ma_{period}']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Volume features
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Remove NaN values
        df = df.dropna()
        
        print(f"Created {df.shape[1]} features, {len(df):,} valid records")
        return df
    
    def prepare_sequences(self, df, sequence_length=60, target_col='close'):
        """Prepare data for time series modeling"""
        print(f"Preparing sequences with length {sequence_length}...")
        
        # Select features for modeling
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 
                       'ma_5', 'ma_20', 'ma_50', 'rsi', 'volatility', 'volume_ratio']
        
        # Scale features
        scaled_data = self.scaler.fit_transform(df[feature_cols])
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i, feature_cols.index(target_col)])  # Close price index
        
        X, y = np.array(X), np.array(y)
        
        print(f"Created {len(X):,} sequences of shape {X.shape}")
        return X, y, feature_cols
    
    def visualize_data(self, df):
        """Create data overview plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Price trend
        axes[0,0].plot(df['close'])
        axes[0,0].set_title('Bitcoin Price Trend')
        axes[0,0].set_ylabel('Price (USDT)')
        
        # Volume
        axes[0,1].plot(df['volume'], alpha=0.7)
        axes[0,1].set_title('Trading Volume')
        axes[0,1].set_ylabel('Volume')
        
        # Returns distribution
        axes[1,0].hist(df['price_change'].dropna(), bins=50, alpha=0.7)
        axes[1,0].set_title('Price Change Distribution')
        axes[1,0].set_xlabel('Daily Return')
        
        # RSI
        axes[1,1].plot(df['rsi'])
        axes[1,1].axhline(70, color='r', linestyle='--', alpha=0.5)
        axes[1,1].axhline(30, color='g', linestyle='--', alpha=0.5)
        axes[1,1].set_title('RSI Indicator')
        axes[1,1].set_ylabel('RSI')
        
        plt.tight_layout()
        plt.savefig('bitcoin_data_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualization saved as bitcoin_data_overview.png")
    
    def generate_report(self, df, X, y, gap_count=0):
        """Generate data preparation report"""
        print("\n" + "="*50)
        print("DATA PREPARATION REPORT")
        print("="*50)
        
        print(f"Raw Data:")
        print(f"   - Records: {len(df):,}")
        print(f"   - Features: {df.shape[1]}")
        print(f"   - Time span: {(df['open_time'].max() - df['open_time'].min()).days} days")
        print(f"   - Time gaps: {gap_count} intervals")
        
        print(f"\nProcessed Data:")
        print(f"   - Training sequences: {len(X):,}")
        print(f"   - Sequence length: {X.shape[1]}")
        print(f"   - Feature dimension: {X.shape[2]}")
        
        print(f"\nPrice Statistics:")
        print(f"   - Min price: ${df['close'].min():.2f}")
        print(f"   - Max price: ${df['close'].max():.2f}")
        print(f"   - Mean price: ${df['close'].mean():.2f}")
        print(f"   - Volatility: {df['close'].std():.2f}")
        
        # Data quality assessment
        completeness = ((len(df) * df.shape[1] - df.isnull().sum().sum()) / (len(df) * df.shape[1]) * 100)
        continuity_score = max(0, 100 - (gap_count / len(df) * 1000))  # Penalize gaps
        overall_quality = (completeness + continuity_score) / 2
        
        print(f"\nData Quality Assessment:")
        print(f"   - Completeness: {completeness:.1f}%")
        print(f"   - Continuity: {continuity_score:.1f}%")
        print(f"   - Overall Quality: {overall_quality:.1f}%")
        
        if overall_quality >= 95:
            print("   Grade: EXCELLENT")
        elif overall_quality >= 85:
            print("   Grade: VERY GOOD")
        else:
            print("   Grade: GOOD")
            
        print("Data ready for Transformer model training!")

def main():
    """Main function"""
    print("Bitcoin Data Preparation")
    print("Course: Data 612 Deep Learning")
    print("="*50)
    
    # Initialize processor
    processor = BitcoinDataProcessor()
    
    # Process data
    file_path = 'btc_1h_data_2018_to_2025.csv'
    
    # Step 1: Load and clean
    df = processor.load_and_clean_data(file_path)
    
    # Step 2: Check time continuity
    gap_count = processor.check_time_gaps(df)
    
    # Step 3: Create features  
    df = processor.create_features(df)
    
    # Step 4: Prepare sequences
    X, y, feature_cols = processor.prepare_sequences(df)
    
    # Step 5: Visualize
    processor.visualize_data(df)
    
    # Step 6: Generate report
    processor.generate_report(df, X, y, gap_count)
    
    # Save processed data and scaler
    df.to_csv('bitcoin_processed.csv', index=False)
    np.save('bitcoin_sequences_X.npy', X)
    np.save('bitcoin_sequences_y.npy', y)
    processor.save_scaler('bitcoin_scaler.pkl')
    
    print(f"\nFiles saved:")
    print(f"   - bitcoin_processed.csv")
    print(f"   - bitcoin_sequences_X.npy") 
    print(f"   - bitcoin_sequences_y.npy")
    print(f"   - bitcoin_scaler.pkl")
    
    return df, X, y, feature_cols

if __name__ == "__main__":
    df, X, y, features = main()