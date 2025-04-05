import sys
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from strategies.LorentzianStrategy.indicators.rsi import RSIIndicator
from strategies.LorentzianStrategy.indicators.wave_trend import WaveTrendIndicator
from strategies.LorentzianStrategy.indicators.cci import CCIIndicator
from strategies.LorentzianStrategy.indicators.adx import ADXIndicator
import time
import gc

# Set up GPU device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Add strategy path to system path
strategy_path = Path(__file__).parent / 'strategies'
sys.path.append(str(strategy_path))

class LorentzianANN:
    """
    Implements a classifier using Approximate Nearest Neighbors with Lorentzian distance,
    similar to TradingView's approach.
    """
    def __init__(
            self,
            lookback_bars=50,      # How many historical bars to consider
            prediction_bars=4,     # How many bars into the future to predict
            k_neighbors=20,        # Number of nearest neighbors to consider
            use_regime_filter=True,
            use_volatility_filter=True,
            use_adx_filter=True,
            adx_threshold=20.0,
            regime_threshold=-0.1
        ):
        self.lookback_bars = lookback_bars
        self.prediction_bars = prediction_bars
        self.k_neighbors = k_neighbors
        self.use_regime_filter = use_regime_filter
        self.use_volatility_filter = use_volatility_filter
        self.use_adx_filter = use_adx_filter
        self.adx_threshold = adx_threshold
        self.regime_threshold = regime_threshold
        
        # These will store our historical data
        self.feature_arrays = None
        self.labels = None
        
    def lorentzian_distance(self, features, historical_features):
        """
        Calculate Lorentzian distance between features and historical features
        
        Args:
            features: torch.Tensor of shape (n_samples, n_features)
            historical_features: torch.Tensor of shape (n_historical, n_features)
            
        Returns:
            distances: torch.Tensor of shape (n_samples, n_historical)
        """
        # Process in batches to save memory
        batch_size = 100  # Adjust based on available memory
        n_samples = features.shape[0]
        n_historical = historical_features.shape[0]
        
        # Initialize distances tensor
        distances = torch.zeros((n_samples, n_historical), device=device)
        
        # Process in batches
        for i in range(0, n_samples, batch_size):
            end_i = min(i + batch_size, n_samples)
            batch_features = features[i:end_i]
            
            # For each feature vector in the batch
            for j in range(batch_features.shape[0]):
                # Get the feature vector
                x = batch_features[j]
                
                # Calculate differences
                diff = torch.abs(x.unsqueeze(0) - historical_features)
                
                # Calculate Lorentzian distance: ln(1 + |x - y|)
                # We use log1p for numerical stability
                # This is the true formula from TradingView
                log_diff = torch.log1p(diff)
                
                # Sum over features dimension
                batch_distances = torch.sum(log_diff, dim=1)
                
                # Store in the distances tensor
                distances[i + j] = batch_distances
                
        return distances
    
    def generate_training_data(self, features, prices):
        """Generate training labels for the prediction task"""
        # Using the same TradingView approach: look ahead a fixed number of bars
        # to determine if price went up or down
        future_prices = prices[self.prediction_bars:]
        current_prices = prices[:-self.prediction_bars]
        
        # Generate labels: 1 for long (price went up), -1 for short (price went down), 0 for neutral
        labels = torch.zeros(len(current_prices), dtype=torch.long)
        labels[future_prices > current_prices] = 1  # Long
        labels[future_prices < current_prices] = -1  # Short
        
        # We can only generate labels for data points that have future data
        return features[:len(labels)], labels
        
    def fit(self, features, prices):
        """Store the training data for ANN lookup"""
        # Convert to tensors if they aren't already
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)
        if not isinstance(prices, torch.Tensor):
            prices = torch.tensor(prices, dtype=torch.float32)
            
        # Generate training data
        features, labels = self.generate_training_data(features, prices)
        
        # Store for future lookup
        self.feature_arrays = features.to(device)
        self.labels = labels.to(device)
        
        return self
    
    def predict(self, features):
        """
        Predict using Approximate Nearest Neighbors with Lorentzian distance
        
        Args:
            features: torch.Tensor of shape (n_samples, n_features)
            
        Returns:
            Predicted labels: 1 for long, -1 for short, 0 for neutral
        """
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)
            
        # Move to same device as model
        features = features.to(device)
        
        # Process in batches to avoid memory issues
        batch_size = 1000  # Adjust based on GPU memory
        n_samples = len(features)
        all_predictions = []
        
        print(f"Processing {n_samples} samples in batches of {batch_size}...")
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_features = features[start_idx:end_idx]
            
            # Calculate Lorentzian distances for this batch
            batch_distances = self.lorentzian_distance(batch_features, self.feature_arrays)
            
            # Get indices of k-nearest neighbors
            _, indices = torch.topk(batch_distances, min(self.k_neighbors, len(batch_distances[0])), 
                                   largest=False, dim=1)
            
            # Get labels of k-nearest neighbors
            batch_neighbor_labels = [self.labels[idx] for idx in indices]
            batch_neighbor_labels = torch.stack(batch_neighbor_labels)
            
            # Calculate the sum of neighbor labels
            batch_predictions = torch.sum(batch_neighbor_labels, dim=1)
            
            # Convert to direction: 1 for long, -1 for short, 0 for neutral
            batch_final = torch.zeros_like(batch_predictions)
            batch_final[batch_predictions > 0] = 1
            batch_final[batch_predictions < 0] = -1
            
            all_predictions.append(batch_final)
            
            # Print progress
            progress = min(100, (end_idx / n_samples) * 100)
            print(f"Progress: {progress:.1f}%", end="\r")
            
        # Combine all batches
        final_predictions = torch.cat(all_predictions)
        print("\nPrediction complete!")
        
        return final_predictions

def prepare_indicators(df):
    """Add technical indicators using custom PyTorch implementations"""
    df = df.copy()
    print(f"\nInitial data shape: {df.shape}")
    
    # Initialize indicators with GPU
    rsi_14 = RSIIndicator(period=14, device=device)
    rsi_9 = RSIIndicator(period=9, device=device)
    wavetrend = WaveTrendIndicator(channel_length=10, average_length=11, device=device)
    cci = CCIIndicator(period=20, device=device)
    adx = ADXIndicator(period=20, device=device)

    # Convert data to tensors and move to GPU
    close = torch.tensor(df['close'].values, dtype=torch.float32).to(device)
    high = torch.tensor(df['high'].values, dtype=torch.float32).to(device)
    low = torch.tensor(df['low'].values, dtype=torch.float32).to(device)
    
    print("\nCalculating indicators...")
    
    try:
        # Calculate indicators
        rsi_14_values = rsi_14.forward(close)['rsi']
        print("RSI-14 calculated successfully")
        
        rsi_9_values = rsi_9.forward(close)['rsi']
        print("RSI-9 calculated successfully")
        
        wt_values = wavetrend.forward(high, low, close)
        print("WaveTrend calculated successfully")
        
        cci_values = cci.forward(high, low, close)['cci']
        print("CCI calculated successfully")
        
        adx_values = adx.forward(high, low, close)['adx']
        print("ADX calculated successfully")
        
        print("\nMoving indicators to CPU...")
        
        # Move tensors back to CPU
        df['rsi_14'] = rsi_14_values.cpu().numpy()
        df['rsi_9'] = rsi_9_values.cpu().numpy()
        df['wt1'] = wt_values['wt1'].cpu().numpy()
        df['wt2'] = wt_values['wt2'].cpu().numpy()
        df['cci'] = cci_values.cpu().numpy()
        df['adx'] = adx_values.cpu().numpy()
        
        print("\nCalculating additional features...")
        
        # Price action features
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['regime'] = df['returns'].rolling(window=50).mean()
        
        # Momentum features
        df['rsi_diff'] = df['rsi_14'] - df['rsi_9']  # RSI divergence
        df['wt_diff'] = df['wt1'] - df['wt2']  # WaveTrend divergence
        
        # Trend features
        df['price_trend'] = df['close'].rolling(window=20).mean().pct_change()
        df['volume_trend'] = df['volume'].rolling(window=20).mean().pct_change()
        
        # Volatility features
        df['tr'] = pd.DataFrame({
            'hl': df['high'] - df['low'],
            'hc': abs(df['high'] - df['close'].shift()),
            'lc': abs(df['low'] - df['close'].shift())
        }).max(axis=1)
        df['atr'] = df['tr'].rolling(window=14).mean()
        df['atr_ratio'] = df['atr'] / df['close']
        
        # Generate labels (1 for profitable trades, 0 for unprofitable)
        # Note: We'll calculate actual labels in the model now
        
        # Drop NaN values
        print("\nRemoving NaN values...")
        df_before = len(df)
        df = df.dropna()
        df_after = len(df)
        print(f"Rows before dropping NaN: {df_before:,}")
        print(f"Rows after dropping NaN: {df_after:,}")
        print(f"Dropped {df_before - df_after:,} rows")
        
    except Exception as e:
        print(f"Error in prepare_indicators: {str(e)}")
        raise
        
    return df

def apply_filters(df, adx_threshold=20.0, regime_threshold=-0.1):
    """Apply filters to identify tradeable conditions (TradingView style)"""
    print(f"\nApplying filters...")
    print(f"Initial filter data shape: {df.shape}")
    
    # Volatility filter - price movement should be significant
    volatility_mask = df['volatility'] > df['volatility'].quantile(0.1)
    print(f"Volatility filter passes: {volatility_mask.sum()}")
    
    # Regime filter - trend should be established (positive regime = uptrend biased)
    regime_mask = df['regime'] > regime_threshold
    print(f"Regime filter passes: {regime_mask.sum()}")
    
    # ADX filter - trend should be strong enough
    adx_mask = df['adx'] > adx_threshold
    print(f"ADX filter passes: {adx_mask.sum()}")
    
    # Combine all filters
    combined_mask = volatility_mask & regime_mask & adx_mask
    print(f"Combined filter passes: {combined_mask.sum()}")
    
    return df[combined_mask]

def prepare_features(df):
    """Prepare features for the model"""
    # Following TradingView's approach with the exact same features
    features = np.column_stack([
        df['rsi_14'].values,  # Feature 1: RSI(14)
        df['wt1'].values,     # Feature 2: WaveTrend
        df['cci'].values,     # Feature 3: CCI
        df['adx'].values,     # Feature 4: ADX
        df['rsi_9'].values    # Feature 5: RSI(9)
    ])
    
    # Scale features to [0, 1] range
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    
    return features, scaler

def train_lorentzian_ann(df, lookback_bars=50, prediction_bars=4, k_neighbors=20):
    """Train the Lorentzian ANN model"""
    try:
        # Prepare features
        features, scaler = prepare_features(df)
        close_prices = df['close'].values
        
        print(f"\nTraining Lorentzian ANN model...")
        print(f"Features shape: {features.shape}")
        print(f"Using {lookback_bars} bars for lookback")
        print(f"Predicting {prediction_bars} bars into the future")
        print(f"Using {k_neighbors} nearest neighbors")
        
        # Free up memory
        gc.collect()
        torch.cuda.empty_cache()
        
        # Initialize model before loading data
        model = LorentzianANN(
            lookback_bars=lookback_bars,
            prediction_bars=prediction_bars,
            k_neighbors=k_neighbors,
            use_regime_filter=True,
            use_volatility_filter=True,
            use_adx_filter=True
        )
        
        # Convert to tensors - use float16 to reduce memory usage
        features_tensor = torch.tensor(features, dtype=torch.float16).to(device)
        prices_tensor = torch.tensor(close_prices, dtype=torch.float16).to(device)
        
        # Fit the model
        model.fit(features_tensor, prices_tensor)
        
        # Free memory before prediction
        gc.collect()
        torch.cuda.empty_cache()
        
        # Generate predictions in batches
        batch_size = 500  # Use smaller batches
        all_predictions = []
        
        for start_idx in range(0, len(features), batch_size):
            end_idx = min(start_idx + batch_size, len(features))
            print(f"Generating predictions for batch {start_idx}-{end_idx}")
            
            # Get batch features
            batch_features = features[start_idx:end_idx]
            batch_tensor = torch.tensor(batch_features, dtype=torch.float16).to(device)
            
            # Get predictions for this batch
            batch_predictions = model.predict(batch_tensor)
            
            # Move to CPU
            all_predictions.append(batch_predictions.cpu().numpy())
            
            # Free memory
            del batch_tensor
            torch.cuda.empty_cache()
        
        # Combine predictions from all batches
        predictions = np.concatenate(all_predictions)
        
        # Store predictions in the dataframe
        df['signal'] = predictions
        
        # Print prediction distribution
        unique, counts = np.unique(predictions, return_counts=True)
        distribution = dict(zip(unique, counts))
        print("\nPrediction distribution:")
        for signal, count in distribution.items():
            signal_name = "LONG" if signal == 1 else "SHORT" if signal == -1 else "NEUTRAL"
            print(f"{signal_name}: {count} ({count/len(predictions)*100:.2f}%)")
            
        return df, model, scaler
        
    except Exception as e:
        print(f"Error in train_lorentzian_ann: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None, None, None

def calculate_metrics(df):
    """Calculate trading metrics"""
    # Calculate returns based on signals
    df['position'] = df['signal'].shift(1).fillna(0)  # Position at the start of the bar
    df['returns'] = df['close'].pct_change()
    df['strategy_returns'] = df['position'] * df['returns']
    
    # Calculate metrics
    total_trades = (df['position'].diff() != 0).sum()
    winning_trades = (df['strategy_returns'] > 0).sum()
    losing_trades = (df['strategy_returns'] < 0).sum()
    
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    avg_win = df.loc[df['strategy_returns'] > 0, 'strategy_returns'].mean() if winning_trades > 0 else 0
    avg_loss = df.loc[df['strategy_returns'] < 0, 'strategy_returns'].mean() if losing_trades > 0 else 0
    
    # Calculate cumulative returns
    df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()
    final_return = df['cumulative_returns'].iloc[-1] if len(df) > 0 else 1.0
    
    # Print metrics
    print("\nPerformance Metrics:")
    print(f"Total Trades: {total_trades}")
    print(f"Winning Trades: {winning_trades}")
    print(f"Losing Trades: {losing_trades}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Average Win: {avg_win:.2%}")
    print(f"Average Loss: {avg_loss:.2%}")
    print(f"Final Return: {final_return - 1:.2%}")
    
    # Calculate max drawdown
    peak = df['cumulative_returns'].cummax()
    drawdown = (df['cumulative_returns'] / peak - 1)
    max_drawdown = drawdown.min()
    print(f"Max Drawdown: {max_drawdown:.2%}")
    
    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'final_return': final_return - 1,
        'max_drawdown': max_drawdown
    }

def plot_results(df):
    """Plot trading signals and cumulative returns"""
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.title('Price with Trading Signals')
    plt.plot(df.index, df['close'], label='Price', alpha=0.7)
    
    # Plot buy and sell signals
    buy_points = df[df['signal'] == 1].index
    sell_points = df[df['signal'] == -1].index
    
    plt.scatter(buy_points, df.loc[buy_points, 'close'], marker='^', color='green', label='Buy Signal')
    plt.scatter(sell_points, df.loc[sell_points, 'close'], marker='v', color='red', label='Sell Signal')
    
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.title('Strategy Cumulative Returns')
    plt.plot(df.index, df['cumulative_returns'], label='Strategy Returns')
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.3)
    plt.ylabel('Cumulative Returns')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('lorentzian_ann_results.png')
    print("Saved plot to lorentzian_ann_results.png")

def main():
    start_time = time.time()
    
    # Load data
    data_path = Path("data/bitget/futures/SOL_USDT_USDT-5m-futures.feather")
    df = pd.read_feather(data_path)
    df['datetime'] = pd.to_datetime(df['date'])
    df.set_index('datetime', inplace=True)
    
    # Prepare indicators
    df = prepare_indicators(df)
    
    # Apply filters (optional, you can skip this to use all data)
    # df = apply_filters(df, adx_threshold=20.0, regime_threshold=-0.1)
    
    # Train model
    df, model, scaler = train_lorentzian_ann(
        df, 
        lookback_bars=50,     # TradingView default
        prediction_bars=4,    # TradingView default
        k_neighbors=20        # Number of neighbors to consider
    )
    
    # Calculate metrics
    metrics = calculate_metrics(df)
    
    # Plot results
    plot_results(df)
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()