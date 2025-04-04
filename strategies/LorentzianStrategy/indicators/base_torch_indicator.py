"""
Base class for PyTorch-based technical indicators
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Union, Optional
from dataclasses import dataclass
from contextlib import nullcontext

@dataclass
class TorchIndicatorConfig:
    """Base configuration for torch indicators"""
    device: Optional[str] = None
    dtype: Optional[torch.dtype] = None
    use_amp: bool = False

class BaseTorchIndicator(nn.Module):
    """Base class for all PyTorch-based indicators"""
    
    def __init__(self, config: TorchIndicatorConfig):
        """Initialize base indicator"""
        super().__init__()
        self.config = config
        
        # Setup device and data type
        self.device = (
            torch.device(config.device) if config.device 
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = config.dtype or torch.float32
        self.use_amp = config.use_amp
        self.scaler = (
            torch.amp.GradScaler('cuda') if self.use_amp else None
        )
        
    def to_tensor(
        self, 
        data: Union[np.ndarray, pd.Series, torch.Tensor]
    ) -> torch.Tensor:
        """Convert input data to tensor"""
        if isinstance(data, torch.Tensor):
            return data.to(device=self.device, dtype=self.dtype)
        elif isinstance(data, pd.Series):
            return torch.tensor(
                data.values, 
                device=self.device, 
                dtype=self.dtype
            )
        else:
            return torch.tensor(
                data, 
                device=self.device, 
                dtype=self.dtype
            )
            
    def ema(self, data: torch.Tensor, length: int) -> torch.Tensor:
        """Calculate EMA using PyTorch"""
        alpha = 2.0 / (length + 1)
        alpha_rev = 1 - alpha
        
        scale = 1/alpha_rev
        n = data.shape[0]
        
        r = torch.arange(n, device=self.device)
        scale_arr = scale**r
        offset = data[0]*alpha_rev**(r+1)
        pw0 = alpha*alpha_rev**(n-1)
        
        mult = data*pw0*scale_arr
        cumsums = mult.flip(0).cumsum(0).flip(0)
        out = offset + cumsums*scale_arr
        return out
        
    def sma(self, data: torch.Tensor, length: int) -> torch.Tensor:
        """Calculate SMA using PyTorch"""
        return nn.functional.avg_pool1d(
            data.view(1, 1, -1), 
            kernel_size=length, 
            stride=1
        ).view(-1)
        
    def rma(self, data: torch.Tensor, length: int) -> torch.Tensor:
        """Calculate RMA (Modified Moving Average) using PyTorch"""
        alpha = 1.0 / length
        return self.ema(data, int(2/alpha - 1))
        
    def populate_indicators(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Populate indicator values in the dataframe"""
        raise NotImplementedError(
            "Subclasses must implement populate_indicators"
        )

    def calculate_signals(
        self, 
        data: pd.DataFrame
    ) -> Dict[str, torch.Tensor]:
        """Calculate indicator signals. Must be implemented by child classes.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary of calculated signals
        """
        raise NotImplementedError(
            "Subclasses must implement calculate_signals"
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for the indicator. Must be implemented by child classes.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary of calculated values
        """
        raise NotImplementedError(
            "Subclasses must implement forward"
        )
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Main calculation method that handles data conversion and processing
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary of calculated values as pandas Series
        """
        try:
            # Convert to tensors and calculate
            with (
                torch.cuda.amp.autocast() 
                if self.use_amp 
                else nullcontext()
            ):
                signals = self.calculate_signals(data)
            
            # Convert back to pandas
            return {
                k: pd.Series(v.cpu().numpy(), index=data.index) 
                for k, v in signals.items()
            }
            
        except Exception as e:
            print(f"Error in indicator calculation: {str(e)}")
            return {} 