"""
Enhanced RSI (Relative Strength Index) Implementation using PyTorch

Features:
- GPU acceleration for faster calculations
- Real-time signal generation
- Configurable parameters
- Advanced signal filtering
- Backtesting metrics
- Debug capabilities
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional, Union
from .base_torch_indicator import BaseTorchIndicator, TorchIndicatorConfig
from contextlib import nullcontext

@dataclass
class RsiMetrics:
    """Container for RSI trading metrics"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0

@dataclass
class RSIConfig(TorchIndicatorConfig):
    """Configuration for RSI indicator"""
    period: int = 14
    overbought: float = 70.0
    oversold: float = 30.0

class RSIIndicator(BaseTorchIndicator):
    """PyTorch-based RSI implementation"""
    
    def __init__(
        self,
        period: int = 14,
        overbought: float = 70.0,
        oversold: float = 30.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        config: Optional[RSIConfig] = None
    ):
        """Initialize RSI indicator with PyTorch backend"""
        if config is None:
            config = RSIConfig(
                period=period,
                overbought=overbought,
                oversold=oversold,
                device=device,
                dtype=dtype
            )
        super().__init__(config)
        self.config = config
        
        # Trading metrics
        self.metrics = RsiMetrics()
        
        # Initialize learnable parameters if needed
        self.alpha = nn.Parameter(torch.tensor(2.0 / (self.config.period + 1)))
        
    @property
    def period(self) -> int:
        return self.config.period
        
    @property
    def overbought(self) -> float:
        return self.config.overbought
        
    @property
    def oversold(self) -> float:
        return self.config.oversold
        
    def forward(self, close: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate RSI values using PyTorch operations
        
        Args:
            close: Close prices tensor
            
        Returns:
            Dictionary with RSI values and signals
        """
        # Calculate price changes
        price_diff = torch.diff(close, dim=0)
        
        # Add padding for the first element
        padding = torch.zeros(1, device=self.device, dtype=self.dtype)
        price_diff = torch.cat([padding, price_diff])

        # Separate gains and losses
        gains = torch.where(price_diff > 0, price_diff, torch.zeros_like(price_diff))
        losses = torch.where(price_diff < 0, -price_diff, torch.zeros_like(price_diff))

        # Calculate average gains and losses
        avg_gains = self.torch_ema(gains, self.alpha.item())
        avg_losses = self.torch_ema(losses, self.alpha.item())

        # Calculate relative strength and RSI
        rs = avg_gains / (avg_losses + 1e-10)  # Add small epsilon to avoid division by zero
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # Generate signals
        buy_signals = torch.zeros_like(rsi, dtype=self.dtype)
        sell_signals = torch.zeros_like(rsi, dtype=self.dtype)

        valid_mask = ~torch.isnan(rsi)
        buy_signals[valid_mask] = (rsi[valid_mask] < self.config.oversold).to(self.dtype)
        sell_signals[valid_mask] = (rsi[valid_mask] > self.config.overbought).to(self.dtype)

        return {
            'rsi': rsi,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals
        }
    
    def calculate_signals(self, data: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """
        Calculate RSI and generate trading signals
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with RSI values and signals
        """
        # Convert price data to tensor
        close_prices = self.to_tensor(data['close'])
        
        # Calculate RSI and signals
        with torch.amp.autocast('cuda') if self.config.use_amp else nullcontext():
            results = self.forward(close_prices)
        
        return results
    
    def update_metrics(self, current_price: float, signal: int, last_signal: int) -> None:
        """Update trading metrics"""
        if last_signal != 0 and signal != last_signal:
            self.metrics.total_trades += 1
            pnl = (current_price - self.last_price) * last_signal
            
            if pnl > 0:
                self.metrics.winning_trades += 1
                self.metrics.avg_win = ((self.metrics.avg_win * 
                    (self.metrics.winning_trades - 1) + pnl) / 
                    self.metrics.winning_trades)
            else:
                self.metrics.losing_trades += 1
                self.metrics.avg_loss = ((self.metrics.avg_loss * 
                    (self.metrics.losing_trades - 1) + abs(pnl)) / 
                    self.metrics.losing_trades)
            
            if self.metrics.total_trades > 0:
                self.metrics.win_rate = (self.metrics.winning_trades / 
                    self.metrics.total_trades)
                
            if self.metrics.avg_loss > 0:
                self.metrics.profit_factor = ((self.metrics.avg_win * 
                    self.metrics.winning_trades) / 
                    (self.metrics.avg_loss * self.metrics.losing_trades))
        
        self.last_price = current_price
    
    def plot_signals(self, df: pd.DataFrame, signals: Dict[str, pd.Series]) -> None:
        """Plot RSI with signals using matplotlib"""
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2, 1])
            
            # Plot price and signals
            ax1.plot(df.index, df['close'], label='Price', alpha=0.7)
            buy_points = df.index[signals['buy_signals'] == 1]
            sell_points = df.index[signals['sell_signals'] == 1]
            
            if len(buy_points) > 0:
                ax1.scatter(buy_points, df.loc[buy_points, 'close'], 
                          color='green', marker='^', label='Buy')
            if len(sell_points) > 0:
                ax1.scatter(sell_points, df.loc[sell_points, 'close'], 
                          color='red', marker='v', label='Sell')
            
            ax1.set_title('Price with RSI Signals')
            ax1.legend()
            
            # Plot RSI
            ax2.plot(df.index, signals['rsi'], label='RSI', color='blue')
            ax2.axhline(y=self.config.overbought, color='r', linestyle='--', alpha=0.5)
            ax2.axhline(y=self.config.oversold, color='g', linestyle='--', alpha=0.5)
            ax2.set_title('RSI')
            ax2.set_ylim(0, 100)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error plotting signals: {str(e)}")
    
    def get_metrics(self) -> Dict:
        """Get current trading metrics"""
        return {
            'total_trades': self.metrics.total_trades,
            'win_rate': self.metrics.win_rate,
            'profit_factor': self.metrics.profit_factor,
            'avg_win': self.metrics.avg_win,
            'avg_loss': self.metrics.avg_loss
        } 