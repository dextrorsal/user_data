# Lorentzian ANN Trading System

This repository contains a machine learning-based trading system that uses a Lorentzian Approximate Nearest Neighbors classifier to generate trading signals for cryptocurrency markets.

## Setup

1. Ensure you have Freqtrade installed. If not, follow the [Freqtrade installation guide](https://www.freqtrade.io/en/stable/installation/).

2. Clone this repository or copy the files to your Freqtrade `user_data` directory.

3. Install required Python packages:
   ```
   pip install torch pandas numpy matplotlib scikit-learn
   ```

## Files

- `analyze_lorentzian_ann.py`: The core implementation of the Lorentzian ANN model
- `test_lorentzian_save.py`: Script to test the model's save/load functionality
- `test_freqtrade_lorentzian.py`: Script to test the model with Freqtrade's backtesting data
- `run_live_trading.py`: Script to start Freqtrade with live trading
- `monitor_trades.py`: Script to monitor trading performance
- `config_live.json`: Configuration file for live trading

## Running Backtests

To run a backtest:

```bash
python test_freqtrade_lorentzian.py --pair SOL/USDT --timeframe 5m
```

This will:
1. Load historical data
2. Either load a pre-trained model or train a new one
3. Generate predictions
4. Calculate performance metrics
5. Plot results

## Live Trading

### Setup

1. Edit `config_live.json` to:
   - Set your exchange API keys (if not using dry-run mode)
   - Configure trading pairs
   - Adjust risk parameters

2. Start the trading bot in dry-run mode:
   ```bash
   python run_live_trading.py
   ```

3. Monitor trading performance:
   ```bash
   python monitor_trades.py
   ```

### Going Live

When ready to trade with real money:

1. Start with a small amount
2. Run:
   ```bash
   python run_live_trading.py --live
   ```

**WARNING**: Live trading uses real money. Use at your own risk!

## Strategy Settings

The Lorentzian ANN strategy has these key parameters:

- `lookback_bars`: Number of bars to look back for training (default: 50)
- `prediction_bars`: Number of bars to look ahead for prediction (default: 4)
- `k_neighbors`: Number of neighbors to consider (default: 20)

You can adjust these in the strategy file or through Freqtrade's hyperopt.

## Model Training

The model is trained on historical data and automatically updates during live trading. It uses:

1. Feature extraction: RSI, WaveTrend, CCI, and ADX indicators
2. Lorentzian distance metric for better pattern recognition
3. Continuous learning to adapt to market changes

## Performance Monitoring

The monitoring script provides:
- Win rate and total profit
- Performance by trading pair
- Open trade status
- Visual performance chart

## Advanced Usage

### Incremental Training

The model supports incremental training to continuously adapt to market conditions:

```python
# Load existing model
model = LorentzianANN()
model.load_model("models/lorentzian_model.pt")

# Update with new data
model.update_model(new_features, new_prices, max_samples=500)
```

### Parameter Optimization

To optimize strategy parameters:

```bash
freqtrade hyperopt --hyperopt-loss SharpeHyperOptLoss --strategy LorentzianStrategy
```

## Archived Files

Previous model implementations and test scripts are stored in the `archived` folder for reference. 