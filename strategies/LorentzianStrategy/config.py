"""
Configuration for the Integrated ML Trading System

This module defines configuration classes and default values for:
1. Lorentzian ANN
2. Logistic Regression
3. Chandelier Exit
4. Risk Management
5. Backtesting
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import torch

@dataclass
class LorentzianConfig:
    """Configuration for Lorentzian ANN model"""
    lookback_bars: int = 50
    prediction_bars: int = 4
    k_neighbors: int = 20
    use_regime_filter: bool = True
    use_volatility_filter: bool = True
    use_adx_filter: bool = True
    adx_threshold: float = 20.0
    regime_threshold: float = -0.1
    features: List[str] = None
    
    def __post_init__(self):
        if self.features is None:
            # Default feature set
            self.features = ['rsi_14', 'wt1', 'wt2', 'cci', 'adx']

@dataclass
class LogisticConfig:
    """Configuration for Logistic Regression model"""
    use_deep: bool = True
    use_amp: bool = False
    learning_rate: float = 0.01
    batch_size: int = 32
    epochs: int = 100
    hidden_size: int = 16
    num_layers: int = 3
    dropout: float = 0.2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    lookback: int = 20
    threshold: float = 0.6
    volatility_filter: bool = True
    volume_filter: bool = True
    input_features: List[str] = None
    
    def __post_init__(self):
        if self.input_features is None:
            # Default feature set for logistic regression
            self.input_features = ['close', 'volume', 'rsi_14', 'wt1', 'adx']

@dataclass
class ChandelierConfig:
    """Configuration for Chandelier Exit"""
    atr_period: int = 22
    atr_multiplier: float = 3.0
    use_close: bool = False
    device: Optional[str] = None
    dtype: Optional[torch.dtype] = None
    use_amp: bool = False

@dataclass
class RiskManagementConfig:
    """Configuration for risk management"""
    max_position_size: float = 0.1  # 10% of account
    risk_per_trade: float = 0.02    # 2% risk per trade
    max_open_trades: int = 3        # Maximum number of concurrent trades
    trailing_stop: bool = True      # Use trailing stops
    take_profit_atr: float = 4.0    # Take profit at 4x ATR
    break_even_atr: float = 2.0     # Move stop to break even at 2x ATR
    scale_out: bool = True          # Scale out of positions
    scale_out_levels: List[float] = None
    
    def __post_init__(self):
        if self.scale_out_levels is None:
            self.scale_out_levels = [0.33, 0.5, 1.0]  # Scale out at 33%, 50%, and 100% of target

@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    initial_capital: float = 10000.0
    commission: float = 0.001       # 0.1% commission
    slippage: float = 0.0005        # 0.05% slippage
    leverage: float = 1.0           # No leverage by default
    data_timeframe: str = "5m"       
    test_period_days: int = 30
    warmup_period_days: int = 30
    use_asymmetric_returns: bool = True  # Weight drawdowns more heavily

@dataclass
class IntegratedSystemConfig:
    """Master configuration for the entire integrated system"""
    lorentzian: LorentzianConfig = LorentzianConfig()
    logistic: LogisticConfig = LogisticConfig()
    chandelier: ChandelierConfig = ChandelierConfig()
    risk: RiskManagementConfig = RiskManagementConfig()
    backtest: BacktestConfig = BacktestConfig()
    model_dir: str = "models"
    log_dir: str = "logs"
    use_gpu: bool = torch.cuda.is_available()
    
    # Signal combination settings
    require_both_signals: bool = True  # Require both models to agree on signal
    lorentzian_weight: float = 0.6     # Weight for Lorentzian signals (0-1)
    logistic_weight: float = 0.4       # Weight for Logistic signals (0-1)

# Default configuration
default_config = IntegratedSystemConfig()

# Configuration for more aggressive trading
aggressive_config = IntegratedSystemConfig(
    lorentzian=LorentzianConfig(
        k_neighbors=10,
        adx_threshold=15.0
    ),
    logistic=LogisticConfig(
        threshold=0.55
    ),
    chandelier=ChandelierConfig(
        atr_multiplier=2.5
    ),
    risk=RiskManagementConfig(
        max_position_size=0.15,
        risk_per_trade=0.03
    ),
    backtest=BacktestConfig(
        leverage=3.0
    ),
    require_both_signals=False
)

# Configuration for conservative trading
conservative_config = IntegratedSystemConfig(
    lorentzian=LorentzianConfig(
        k_neighbors=30,
        adx_threshold=25.0
    ),
    logistic=LogisticConfig(
        threshold=0.7
    ),
    chandelier=ChandelierConfig(
        atr_multiplier=4.0
    ),
    risk=RiskManagementConfig(
        max_position_size=0.05,
        risk_per_trade=0.01
    ),
    require_both_signals=True
) 