# 📁 Project Structure: Lorentzian Trading System

This document provides a clear overview of the project structure and explains how all components fit together.

## 📋 Component Overview

The trading system is composed of three main components:

1. **🔍 Primary Signal Generator: Lorentzian ANN**
   - Located in `models/primary/lorentzian_classifier.py`
   - Generates initial trading signals using K-Nearest Neighbors with Lorentzian distance

2. **✅ Signal Confirmation: Logistic Regression**
   - Located in `models/confirmation/logistic_regression_torch.py`
   - Validates signals from the primary generator to reduce false positives

3. **🛡️ Risk Management: Chandelier Exit**
   - Located in `models/risk_management/chandelier_exit.py` 
   - Manages position exits and stop loss levels based on ATR

## 🔄 Flow of Execution

The typical flow through the system is:

1. Price data is received and indicators are calculated
2. Lorentzian ANN generates potential trading signals
3. Logistic Regression confirms or rejects these signals
4. Chandelier Exit provides stop loss levels and exit signals
5. If all conditions align, a trade is executed

## 📂 Directory Structure

```
strategies/LorentzianStrategy/
│
├── indicators/                  # Technical indicators with PyTorch acceleration
│   ├── base_torch_indicator.py  # Base class for all indicators
│   ├── rsi.py                   # RSI implementation
│   ├── cci.py                   # CCI implementation
│   ├── adx.py                   # ADX implementation
│   ├── wave_trend.py            # WaveTrend indicator
│   └── chandelier_exit.py       # Chandelier Exit (redundant with risk_management version)
│
├── models/                      # ML model implementations
│   ├── primary/                 # Primary signal generation
│   │   └── lorentzian_classifier.py  # Lorentzian ANN implementation
│   ├── confirmation/            # Signal confirmation
│   │   └── logistic_regression_torch.py  # Logistic Regression model
│   └── risk_management/         # Risk management
│       └── chandelier_exit.py   # Chandelier Exit implementation
│
├── lorentzian_strategy.py       # Main strategy implementation for Freqtrade
├── lorentzian_classifier.py     # Standalone Lorentzian classifier (redundant)
├── integrated_ml_trader.py      # Main integration class combining all components
├── config.py                    # Configuration settings for all components
└── README.md                    # Project documentation
```

## 🔌 Integration Points

- **integrated_ml_trader.py** - Main class that integrates all three components
- **lorentzian_strategy.py** - Freqtrade strategy implementation
- **config.py** - Central configuration for all components

## 🔄 Redundant Files

Some files are redundant and serve similar purposes:

1. **Lorentzian Classifier implementations**:
   - `lorentzian_classifier.py` (standalone)
   - `models/primary/lorentzian_classifier.py` (integrated version)

2. **Chandelier Exit implementations**:
   - `indicators/chandelier_exit.py` (indicator version)
   - `models/risk_management/chandelier_exit.py` (risk management version)

## 📊 How to Use

For standalone backtesting and analysis:
```python
from strategies.LorentzianStrategy.integrated_ml_trader import IntegratedMLTrader

# Create instance
trader = IntegratedMLTrader()

# Load data and calculate indicators
df = prepare_data()

# Train models
trader.train_models(df)

# Generate signals
signals = trader.generate_signals(df)

# Backtest
results = trader.backtest(signals)

# Visualize
trader.plot_results(results)
```

For use with Freqtrade:
```python
# Import the wrapper
from strategies.lorentzian_strategy import LorentzianStrategy

# Freqtrade will handle the instantiation and execution
```

## 🛠️ Development Workflow

1. Implement or modify indicators in the `indicators/` directory
2. Update model implementations in the `models/` directory 
3. Configure parameters in `config.py`
4. Integrate components in `integrated_ml_trader.py`
5. Test with standalone scripts before deploying to Freqtrade

## 📈 Next Steps

To create a fully integrated system with all three components, focus on:

1. Ensuring all components follow consistent API patterns
2. Using the configuration objects from `config.py`
3. Testing each component individually before integration
4. Verifying signal flow through the entire system

---

*This document provides a high-level overview of the system structure. For implementation details, refer to the individual component files and documentation.* 