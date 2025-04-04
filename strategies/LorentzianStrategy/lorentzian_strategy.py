import logging
from functools import reduce
from datetime import datetime
import numpy as np
import torch
from pandas import DataFrame

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from freqtrade.persistence import Trade

# Import your custom components
from .models.torch_model import TradingModel, ModelConfig
from .indicators.wave_trend import WaveTrendIndicator
from .indicators.rsi import RSIIndicator
from .indicators.cci import CCIIndicator
from .indicators.chandelier_exit import ChandelierExitIndicator
from .indicators.adx import ADXIndicator

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
    lookback_period = IntParameter(20, 40, default=30, space="buy", optimize=True)
    confidence_threshold = DecimalParameter(0.6, 0.9, default=0.7, space="buy", optimize=True)
    
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        
        # Initialize PyTorch model
        model_config = ModelConfig(
            input_size=20,
            hidden_size=64,
            num_layers=2,
            dropout=0.2
        )
        self.model = TradingModel(model_config)
        
        # Load model weights if available
        try:
            self.model.load_state_dict(torch.load(
                "user_data/models/lorentzian_model.pth",
                map_location=self.model.device
            ))
            logger.info("Loaded existing model weights")
        except FileNotFoundError:
            logger.warning("No pre-trained model found, using untrained model")
        
        # Initialize indicators
        self.wave_trend = WaveTrendIndicator()
        self.rsi = RSIIndicator()
        self.cci = CCIIndicator()
        self.chandelier = ChandelierExitIndicator()
        self.adx = ADXIndicator()
        
        # Cache for feature data
        self._feature_cache = {}
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate all technical indicators and model predictions"""
        
        # Calculate basic technical indicators
        dataframe = self.wave_trend.populate_indicators(dataframe)
        dataframe = self.rsi.populate_indicators(dataframe)
        dataframe = self.cci.populate_indicators(dataframe)
        dataframe = self.adx.populate_indicators(dataframe)
        
        # Prepare features for the model
        features = self.prepare_features(dataframe)
        
        # Get model predictions
        if len(features) > 0:
            with torch.no_grad():
                direction_logits, confidence = self.model(features)
                
                # Convert predictions to numpy
                direction_probs = torch.softmax(direction_logits, dim=1).cpu().numpy()
                confidence = confidence.cpu().numpy()
                
                # Add predictions to dataframe
                dataframe['dl_long_prob'] = direction_probs[:, 0]
                dataframe['dl_short_prob'] = direction_probs[:, 1]
                dataframe['dl_neutral_prob'] = direction_probs[:, 2]
                dataframe['dl_confidence'] = confidence
        
        # Calculate Chandelier Exit stops
        long_stops, short_stops = self.chandelier.calculate_stops(dataframe)
        dataframe['long_stop'] = long_stops
        dataframe['short_stop'] = short_stops
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Generate entry signals"""
        
        conditions_long = [
            # Strong long signal from model
            (dataframe['dl_long_prob'] > dataframe['dl_short_prob']),
            (dataframe['dl_long_prob'] > dataframe['dl_neutral_prob']),
            (dataframe['dl_confidence'] > self.confidence_threshold.value),
            
            # Confirm with technical indicators
            (dataframe['rsi'] < 70),  # Not overbought
            (dataframe['adx'] > 25),  # Strong trend
            (dataframe['wave_trend'] > 0),  # Bullish wave trend
        ]
        
        conditions_short = [
            # Strong short signal from model
            (dataframe['dl_short_prob'] > dataframe['dl_long_prob']),
            (dataframe['dl_short_prob'] > dataframe['dl_neutral_prob']),
            (dataframe['dl_confidence'] > self.confidence_threshold.value),
            
            # Confirm with technical indicators
            (dataframe['rsi'] > 30),  # Not oversold
            (dataframe['adx'] > 25),  # Strong trend
            (dataframe['wave_trend'] < 0),  # Bearish wave trend
        ]
        
        dataframe.loc[
            reduce(lambda x, y: x & y, conditions_long),
            'enter_long'
        ] = 1
        
        dataframe.loc[
            reduce(lambda x, y: x & y, conditions_short),
            'enter_short'
        ] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Generate exit signals"""
        
        # Exit long positions
        dataframe.loc[
            (
                (dataframe['dl_short_prob'] > dataframe['dl_long_prob']) |
                (dataframe['close'] < dataframe['long_stop'])
            ),
            'exit_long'
        ] = 1
        
        # Exit short positions
        dataframe.loc[
            (
                (dataframe['dl_long_prob'] > dataframe['dl_short_prob']) |
                (dataframe['close'] > dataframe['short_stop'])
            ),
            'exit_short'
        ] = 1
        
        return dataframe
    
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """Dynamic stoploss using Chandelier Exit"""
        
        # Get dataframe for this pair
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1]
        
        # Use Chandelier Exit stops
        if trade.is_short:
            return (last_candle['short_stop'] - current_rate) / current_rate
        
        return (current_rate - last_candle['long_stop']) / current_rate
    
    def prepare_features(self, dataframe: DataFrame) -> torch.Tensor:
        """Prepare features for the PyTorch model"""
        
        # Select feature columns
        feature_columns = [
            'rsi', 'wave_trend', 'cci', 'adx',
            'close', 'volume', 'high', 'low'
        ]
        
        # Create sequences
        sequences = []
        for i in range(len(dataframe) - self.lookback_period.value + 1):
            sequence = dataframe[feature_columns].iloc[i:i+self.lookback_period.value].values
            sequences.append(sequence)
        
        if len(sequences) == 0:
            return torch.tensor([])
        
        # Convert to tensor and normalize
        features = torch.tensor(sequences, dtype=torch.float32)
        features = (features - features.mean()) / (features.std() + 1e-8)
        
        return features.to(self.model.device)
    
    def bot_start(self, **kwargs) -> None:
        """Called when bot starts"""
        logger.info("Lorentzian Strategy starting...")
    
    def bot_cleanup(self) -> None:
        """Called when bot stops"""
        # Clear cache
        self._feature_cache.clear() 