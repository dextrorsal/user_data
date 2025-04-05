# 🚀 Integrated ML Trading System

This system combines multiple machine learning and technical analysis approaches into a cohesive trading strategy. The three core components work together to generate high-quality trading signals while managing risk effectively.

## 🧩 Core Components

### 1. Lorentzian ANN (Primary Signal Generator)
Uses K-Nearest Neighbors with Lorentzian distance metrics to identify trading opportunities based on historical patterns. This approach is similar to TradingView's Lorentzian classification system and serves as our primary signal generator.

### 2. Logistic Regression (Signal Confirmation)
Acts as a confirmation mechanism, using probability-based signal validation to reduce false positives. The deep learning mode provides more sophisticated pattern recognition capabilities.

### 3. Chandelier Exit (Risk Management)
Implements ATR-based trailing stops to manage positions and protect profits. This technique dynamically adjusts stop loss levels based on market volatility.

## 📊 Signal Flow

The system follows this signal generation hierarchy:

```
Lorentzian ANN (Primary Signal) → Logistic Regression (Confirmation) → Chandelier Exit (Risk Management)
```

A trade is only executed when:
1. The Lorentzian ANN generates a signal
2. The Logistic Regression model confirms the signal
3. The Chandelier Exit provides appropriate stop levels

## 🛠️ Technical Features

- **GPU Acceleration**: All components support GPU acceleration for faster training and inference
- **Persistence**: Models can be saved and loaded for later use
- **Configurable Parameters**: Extensive configuration options for all components
- **Adaptive Learning**: Models can be updated with new data without full retraining
- **Risk Management**: Advanced position sizing and stop loss management

## 📈 Trading Strategy

### Entry Criteria
- Long: Lorentzian signal = 1, Logistic probability > threshold
- Short: Lorentzian signal = -1, Logistic probability < (1-threshold)

### Exit Criteria
- Trailing stop hit (Chandelier Exit)
- Opposing signal generated
- Take profit target reached

### Risk Management
- Position sizing based on ATR and account risk parameters
- Trailing stops adjusted as trade moves into profit
- Optional scale-out at predefined profit targets

## 🔧 Configuration

Three primary configuration profiles are available:
- `default_config`: Balanced risk-reward approach
- `aggressive_config`: Higher returns with increased drawdowns
- `conservative_config`: Lower returns with reduced drawdowns

## 📊 Performance Metrics

The system tracks:
- Win rate
- Profit factor
- Average win/loss
- Maximum drawdown
- Sharpe ratio
- Equity curve

## 📖 Usage

```python
# Example usage
from strategies.LorentzianStrategy.integrated_ml_trader import IntegratedMLTrader
from strategies.LorentzianStrategy.config import default_config

# Load data
df = pd.read_feather("data/btc_usdt_5m.feather")

# Add indicators
# ... preprocessing code ...

# Create trader
trader = IntegratedMLTrader(config=default_config)

# Train models
trader.train_models(df)

# Generate signals
df = trader.generate_signals(df)

# Backtest
results = trader.backtest(df)

# Plot results
trader.plot_results(results)
```

## 📦 Files

- `integrated_ml_trader.py`: Main integration class
- `lorentzian_classifier.py`: Lorentzian ANN implementation
- `models/confirmation/logistic_regression_torch.py`: Logistic Regression model
- `models/risk_management/chandelier_exit.py`: Chandelier Exit implementation
- `config.py`: Configuration classes and presets

## 🔍 Future Enhancements

- [ ] Market regime detection
- [ ] Real-time signal generation
- [ ] Multi-timeframe analysis
- [ ] Portfolio optimization
- [ ] Event-based backtesting
- [ ] Web dashboard

---

*This trading system is for educational purposes only. Always conduct your own research and risk assessment before trading.* 