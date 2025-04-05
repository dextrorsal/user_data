#!/usr/bin/env python3
"""
Script to test imports and identify issues
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import logging
from pathlib import Path

print("Basic imports successful")

# Test numpy.nan
print(f"numpy.nan exists: {hasattr(np, 'nan')}")

# Try to import freqtrade components
try:
    from freqtrade.strategy import IStrategy
    print("Freqtrade strategy import successful")
except Exception as e:
    print(f"Freqtrade strategy import failed: {e}")

# Try to import TA libraries
try:
    import talib.abstract as ta
    print("TA-Lib abstract import successful")
    print(f"TA-Lib has RSI: {hasattr(ta, 'RSI')}")
except Exception as e:
    print(f"TA-Lib import failed: {e}")

try:
    import pandas_ta as pta
    print("Pandas TA import successful")
except Exception as e:
    print(f"Pandas TA import failed: {e}")

try:
    from technical import qtpylib
    print("Technical qtpylib import successful")
except Exception as e:
    print(f"Technical qtpylib import failed: {e}")

# Check our strategy directory structure
print("\nChecking strategy directory structure")
user_data_dir = Path("/home/dex/user_data")
strategy_dir = user_data_dir / "strategies"
lorentzian_dir = strategy_dir / "LorentzianStrategy"

print(f"user_data exists: {user_data_dir.exists()}")
print(f"strategies exists: {strategy_dir.exists()}")
print(f"LorentzianStrategy exists: {lorentzian_dir.exists()}")

if lorentzian_dir.exists():
    print("Files in LorentzianStrategy directory:")
    for file in lorentzian_dir.iterdir():
        print(f"  {file.name}")

indicators_dir = lorentzian_dir / "indicators"
if indicators_dir.exists():
    print("Files in indicators directory:")
    for file in indicators_dir.iterdir():
        print(f"  {file.name}")

# Try to import our Lorentzian strategy
print("\nAttempting to import LorentzianStrategy")
sys.path.append(str(strategy_dir))
try:
    from LorentzianStrategy import LorentzianStrategy
    print("LorentzianStrategy import successful")
except Exception as e:
    print(f"LorentzianStrategy import failed: {e}")
    print("Error details:")
    import traceback
    traceback.print_exc() 