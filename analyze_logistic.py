import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import talib
from datetime import datetime

class LogisticRegressionTrader:
    def __init__(
        self,
        lookback: int = 3,
        learning_rate: float = 0.0009,
        iterations: int = 1000,
        holding_period: int = 5,
        use_synthetic: bool = True,
        filter_signals: str = 'Both'  # 'Volatility', 'Volume', 'Both', 'None'
    ):
        self.lookback = lookback
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.holding_period = holding_period
        self.use_synthetic = use_synthetic
        self.filter_signals = filter_signals
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = MinMaxScaler()
        
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features and technical indicators"""
        # Calculate basic price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log1p(df['returns'])
        
        # Generate synthetic dataset as in Pine Script
        if self.use_synthetic:
            df['synthetic'] = np.log(np.abs(np.power(df['close'], 2) - 1) + 0.5)
        
        # Calculate technical indicators
        df['rsi'] = talib.RSI(df['close'])
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'])
        df['volume_rsi'] = talib.RSI(df['volume'])
        df['volume_sma'] = talib.SMA(df['volume'], timeperiod=20)
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df
    
    def apply_filters(self, df: pd.DataFrame) -> pd.Series:
        """Apply volatility and volume filters"""
        volatility_filter = True
        volume_filter = True
        
        if self.filter_signals in ['Volatility', 'Both']:
            atr1 = talib.ATR(df['high'], df['low'], df['close'], timeperiod=1)
            atr10 = talib.ATR(df['high'], df['low'], df['close'], timeperiod=10)
            volatility_filter = atr1 > atr10
            
        if self.filter_signals in ['Volume', 'Both']:
            volume_filter = df['volume_rsi'] > 49
            
        if self.filter_signals == 'None':
            return pd.Series(True, index=df.index)
            
        return volatility_filter & volume_filter
    
    def sigmoid(self, z: torch.Tensor) -> torch.Tensor:
        """Apply sigmoid activation"""
        return 1.0 / (1.0 + torch.exp(-z))
    
    def logistic_regression(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Train logistic regression model"""
        w = torch.zeros(X.shape[1], device=self.device, requires_grad=True)
        optimizer = optim.Adam([w], lr=self.learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        
        for _ in range(self.iterations):
            optimizer.zero_grad()
            z = torch.matmul(X, w)
            pred = self.sigmoid(z)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
        with torch.no_grad():
            final_pred = self.sigmoid(torch.matmul(X, w))
            
        return loss, final_pred
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals"""
        print("Preparing data...")
        df = self.prepare_data(df)
        
        # Create features and target
        X = torch.tensor(df['close'].values, dtype=torch.float32, device=self.device)
        y = torch.tensor(df['synthetic'].values if self.use_synthetic else df['returns'].values,
                        dtype=torch.float32, device=self.device)
        
        # Apply rolling window
        signals = pd.Series(index=df.index, dtype=float)
        hp_counter = pd.Series(0, index=df.index)
        current_signal = 0
        
        print("Generating signals...")
        for i in range(self.lookback, len(df)):
            X_window = X[i-self.lookback:i]
            y_window = y[i-self.lookback:i]
            
            # Train model on window
            loss, pred = self.logistic_regression(X_window.unsqueeze(1), y_window)
            
            # Scale predictions to price range
            scaled_loss = self.scaler.fit_transform(loss.cpu().numpy().reshape(-1, 1)).flatten()[-1]
            scaled_pred = self.scaler.transform(pred.cpu().numpy().reshape(-1, 1)).flatten()[-1]
            
            # Generate signal
            if df['close'].iloc[i] < scaled_loss and self.apply_filters(df).iloc[i]:
                new_signal = -1  # SELL
            elif df['close'].iloc[i] > scaled_loss and self.apply_filters(df).iloc[i]:
                new_signal = 1   # BUY
            else:
                new_signal = current_signal
                
            # Apply holding period logic
            if new_signal != current_signal:
                hp_counter.iloc[i] = 0
            else:
                hp_counter.iloc[i] = hp_counter.iloc[i-1] + 1
                
            if hp_counter.iloc[i] >= self.holding_period and current_signal != 0:
                new_signal = 0
                
            signals.iloc[i] = new_signal
            current_signal = new_signal
            
            if i % 100 == 0:
                print(f"Processing bar {i}/{len(df)}")
        
        df['signal'] = signals
        
        return df
    
    def calculate_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate trading metrics"""
        df['pnl'] = df['returns'] * df['signal'].shift(1)
        df['cumulative_returns'] = (1 + df['pnl']).cumprod()
        
        total_trades = len(df[df['signal'] != df['signal'].shift(1)]) - 1
        winning_trades = len(df[df['pnl'] > 0])
        
        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'final_return': df['cumulative_returns'].iloc[-1] - 1,
            'max_drawdown': (df['cumulative_returns'] / df['cumulative_returns'].cummax() - 1).min()
        }
        
        return metrics
    
    def plot_results(self, df: pd.DataFrame):
        """Plot trading signals and cumulative returns"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot price and signals
        ax1.plot(df.index, df['close'], label='Price', alpha=0.7)
        buy_points = df[df['signal'] == 1].index
        sell_points = df[df['signal'] == -1].index
        
        ax1.scatter(buy_points, df.loc[buy_points, 'close'], 
                   color='green', marker='^', label='Buy')
        ax1.scatter(sell_points, df.loc[sell_points, 'close'],
                   color='red', marker='v', label='Sell')
        
        ax1.set_title('Price with Logistic Regression Signals')
        ax1.legend()
        
        # Plot cumulative returns
        ax2.plot(df.index, df['cumulative_returns'], label='Cumulative Returns')
        ax2.set_title('Cumulative Returns')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('logistic_results.png')
        plt.close()

def main():
    # Load your data
    print("Loading data...")
    df = pd.read_csv('your_data.csv')  # Replace with your data loading
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    
    # Initialize and run strategy
    trader = LogisticRegressionTrader(
        lookback=3,
        learning_rate=0.0009,
        iterations=1000,
        holding_period=5,
        use_synthetic=True,
        filter_signals='Both'
    )
    
    # Generate signals
    df = trader.generate_signals(df)
    
    # Calculate and print metrics
    metrics = trader.calculate_metrics(df)
    print("\nTrading Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.2%}")
    
    # Plot results
    trader.plot_results(df)
    print("\nResults have been plotted to 'logistic_results.png'")

if __name__ == "__main__":
    main()