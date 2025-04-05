import logging
from functools import reduce
from datetime import datetime
import numpy as np
import torch
from pandas import DataFrame
import os
from pathlib import Path
import sys
import pandas as pd

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from freqtrade.persistence import Trade

# Import the Lorentzian ANN model
sys.path.append(str(Path(__file__).parent.parent.parent))  # Add user_data to path
from analyze_lorentzian_ann import LorentzianANN, prepare_indicators

logger = logging.getLogger(__name__)

class LorentzianStrategy(IStrategy):
    INTERFACE_VERSION = 3
    
    # Strategy parameters
    minimal_roi = {
        "0": 0.05,    # 5% minimum ROI
        "30": 0.025,  # 2.5% after 30 minutes
        "60": 0.01,   # 1% after 1 hour
        "120": 0      # Exit after 2 hours regardless of profit
    }
    
    stoploss = -0.1  # 10% stop loss
    
    # Trailing stoploss
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.01
    trailing_only_offset_is_reached = True

    # Timeframe
    timeframe = '5m'
    
    # Hyperopt parameters
    lookback_bars = IntParameter(30, 60, default=50, space="buy", optimize=True)
    prediction_bars = IntParameter(2, 6, default=4, space="buy", optimize=True)
    k_neighbors = IntParameter(10, 30, default=20, space="buy", optimize=True)
    
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize Lorentzian ANN model
        self.model = LorentzianANN(
            lookback_bars=self.lookback_bars.value,
            prediction_bars=self.prediction_bars.value,
            k_neighbors=self.k_neighbors.value,
            use_regime_filter=True,
            use_volatility_filter=True,
            use_adx_filter=True
        )
        
        # Load model weights if available
        model_path = Path("models/lorentzian_model.pt")
        if os.path.exists(model_path):
            try:
                self.model.load_model(model_path)
                logger.info(f"Loaded existing model from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model: {str(e)}")
        else:
            logger.warning(f"No pre-trained model found at {model_path}")
        
        # Cache for indicator data
        self._indicator_cache = {}
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate all technical indicators and model predictions"""
        
        pair = metadata['pair']
        
        # Deep copy dataframe to avoid modifying the original
        df = dataframe.copy()
        
        # Add date/time columns for compatibility
        df['date'] = df['date'].astype(np.int64) // 1000000  # Convert to UNIX timestamp
        df['datetime'] = pd.to_datetime(df['date'], unit='s')
        
        # Calculate indicators
        try:
            # Check if we have calculated indicators for this pair recently
            if pair in self._indicator_cache:
                df_with_indicators = self._indicator_cache[pair]
                # If we have more data, just calculate for the new data
                if len(df) > len(df_with_indicators):
                    new_data = df.iloc[len(df_with_indicators):].copy()
                    new_data_with_indicators = prepare_indicators(new_data)
                    df_with_indicators = pd.concat([df_with_indicators, new_data_with_indicators])
                    self._indicator_cache[pair] = df_with_indicators
            else:
                # Calculate indicators for all data
                df_with_indicators = prepare_indicators(df)
                self._indicator_cache[pair] = df_with_indicators
            
            # Add model predictions
            if self.model.is_fitted:
                # Prepare features for prediction
                feature_cols = ['rsi_14', 'wt1', 'wt2', 'cci', 'adx']
                features = df_with_indicators[feature_cols].values
                
                # Scale features
                if self.model.scaler:
                    scaled_features = np.zeros((len(features), len(feature_cols)))
                    for i, col in enumerate(feature_cols):
                        if col in self.model.scaler:
                            mean = self.model.scaler[col]['mean']
                            std = self.model.scaler[col]['std']
                            scaled_features[:, i] = (features[:, i] - mean) / (std if std > 0 else 1)
                else:
                    # Simple standardization if no scaler available
                    scaled_features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
                
                # Convert to tensor
                features_tensor = torch.tensor(scaled_features, dtype=torch.float32).to(self.device)
                
                # Get predictions
                with torch.no_grad():
                    predictions = self.model.predict(features_tensor)
                
                # Add predictions to dataframe
                dataframe['signal'] = predictions.cpu().numpy()
            else:
                # No model available, use neutral signal
                dataframe['signal'] = 0
                
        except Exception as e:
            logger.error(f"Error calculating indicators/predictions: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            dataframe['signal'] = 0
            
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Generate entry signals"""
        
        # Long entries
        dataframe.loc[
            (dataframe['signal'] == 1),  # Model predicts long
            'enter_long'
        ] = 1
        
        # Short entries
        dataframe.loc[
            (dataframe['signal'] == -1),  # Model predicts short
            'enter_short'
        ] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Generate exit signals"""
        
        # Exit long when model predicts short
        dataframe.loc[
            (dataframe['signal'] == -1),
            'exit_long'
        ] = 1
        
        # Exit short when model predicts long
        dataframe.loc[
            (dataframe['signal'] == 1),
            'exit_short'
        ] = 1
        
        return dataframe
    
    def bot_start(self, **kwargs) -> None:
        """Called when bot starts"""
        logger.info("Lorentzian Strategy starting...")
    
    def bot_cleanup(self) -> None:
        """Called when bot stops"""
        # Clear cache
        self._indicator_cache.clear() 