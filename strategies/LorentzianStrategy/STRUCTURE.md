# 📁 Project Structure: Lorentzian Trading System

This document provides a comprehensive guide to the project structure and explains how all components work together.

## 🎯 System Overview

The trading system combines three core components with supporting infrastructure:

### Core Components

1. **🔍 Primary Signal Generator: Lorentzian ANN**
   - Uses K-Nearest Neighbors with Lorentzian distance
   - Identifies potential trading opportunities
   - Generates initial buy/sell signals

2. **✅ Signal Confirmation: Logistic Regression**
   - Validates signals from the Lorentzian ANN
   - Reduces false positives
   - Provides probability scores for trades

3. **🛡️ Risk Management: Chandelier Exit**
   - Manages position exits
   - Calculates dynamic stop-loss levels
   - Uses ATR for volatility-based adjustments

## 📂 Directory Structure & Component Guide

```
strategies/LorentzianStrategy/
│
├── 📊 Core Strategy Files
│   ├── lorentzian_strategy.py     # Main Freqtrade strategy implementation
│   ├── integrated_ml_trader.py    # Combines all ML components
│   ├── generate_signals.py        # Signal generation utilities
│   └── config.py                  # Central configuration
│
├── 📈 Models
│   ├── primary/
│   │   └── lorentzian_classifier.py    # Primary signal generation
│   ├── confirmation/
│   │   └── logistic_regression_torch.py # Signal validation
│   ├── risk_management/
│   │   └── chandelier_exit.py          # Exit management
│   └── torch_model.py                   # Base PyTorch model utilities
│
├── 📉 Indicators
│   ├── base_torch_indicator.py    # Base class for all indicators
│   ├── technical_indicators.py    # Collection of basic indicators
│   ├── trend_levels.py           # Support/Resistance detection
│   ├── rsi.py                    # Relative Strength Index
│   ├── cci.py                    # Commodity Channel Index
│   ├── adx.py                    # Average Directional Index
│   ├── wave_trend.py             # WaveTrend oscillator
│   └── chandelier_exit.py        # Chandelier Exit indicator
│
├── 🧪 Testing & Examples
│   ├── run_backtest.py          # Standalone backtesting script
│   ├── test_lorentzian_save.py  # Model saving/loading tests
│   └── examples/                 # Usage examples and notebooks
│
└── 📚 Documentation
    ├── README.md                # Quick start guide
    ├── DOCUMENTATION.md         # Detailed component documentation
    ├── INTEGRATION.md          # Integration guidelines
    ├── STRUCTURE.md            # This file
    └── requirements.txt        # Project dependencies
```

## 🔄 Data Flow & Integration

### Signal Generation Pipeline
1. Raw price data → Technical Indicators
2. Indicators → Lorentzian ANN
3. ANN Signals → Logistic Regression
4. Confirmed Signals → Risk Management
5. Final Decisions → Trade Execution

### Key Integration Points

1. **Data Preparation**
   ```python
   from strategies.LorentzianStrategy.indicators.technical_indicators import calculate_indicators
   
   # Prepare data with all required indicators
   data = calculate_indicators(price_data)
   ```

2. **Signal Generation**
   ```python
   from strategies.LorentzianStrategy.generate_signals import generate_trading_signals
   
   # Generate signals using all models
   signals = generate_trading_signals(data)
   ```

3. **Risk Management**
   ```python
   from strategies.LorentzianStrategy.models.risk_management.chandelier_exit import ChandelierExit
   
   # Set up risk management
   risk_manager = ChandelierExit()
   exit_signals = risk_manager.calculate_exits(data, signals)
   ```

## 🛠️ Common Usage Patterns

### 1. Standalone Backtesting
```python
from strategies.LorentzianStrategy.integrated_ml_trader import IntegratedMLTrader

# Initialize the integrated system
trader = IntegratedMLTrader()

# Load and prepare data
data = trader.prepare_data(your_data)

# Run backtest
results = trader.run_backtest(data)
```

### 2. Live Trading with Freqtrade
```python
# In your freqtrade config:
"strategy": "LorentzianStrategy",
"strategy_path": "/path/to/strategies/"
```

### 3. Model Training
```python
from strategies.LorentzianStrategy.models.primary.lorentzian_classifier import LorentzianClassifier
from strategies.LorentzianStrategy.models.confirmation.logistic_regression_torch import LogisticRegressionModel

# Train primary model
lorentzian = LorentzianClassifier()
lorentzian.train(training_data)

# Train confirmation model
confirmation = LogisticRegressionModel()
confirmation.train(training_data)
```

## 🔧 Configuration

Key configuration files and their purposes:

1. `config.py`: Central configuration for all components
   - Model parameters
   - Indicator settings
   - Trading thresholds

2. `requirements.txt`: Project dependencies
   - PyTorch
   - Technical analysis libraries
   - Data processing utilities

## 📈 Development Workflow

1. **Setup**
   ```bash
   pip install -r requirements.txt
   ```

2. **Testing Changes**
   - Use `run_backtest.py` for quick validation
   - Check signal generation with `generate_signals.py`
   - Verify model saving/loading with `test_lorentzian_save.py`

3. **Integration**
   - Follow guidelines in `INTEGRATION.md`
   - Test components individually
   - Verify full system integration

## 🔍 Troubleshooting Guide

Common issues and solutions:

1. **Model Loading Errors**
   - Verify model file paths in `config.py`
   - Check PyTorch version compatibility

2. **Signal Generation Issues**
   - Confirm all required indicators are calculated
   - Verify data preprocessing steps

3. **Performance Problems**
   - Check indicator calculation efficiency
   - Verify GPU utilization if available
   - Monitor memory usage with large datasets

---

*For detailed implementation information, refer to individual component documentation and docstrings within each file.* 