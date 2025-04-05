"""
TEST COMPONENT: Lorentzian ANN Model Persistence Testing

This script provides a comprehensive test suite for the Lorentzian ANN model's
save/load functionality. It verifies that model persistence works correctly and
that the model state can be properly preserved and restored for continuous learning.

Key features:
- Complete implementation of the Lorentzian ANN model with save/load capabilities
- Feature preparation and indicator calculation
- Training data generation and model fitting
- Progressive/incremental learning simulation
- Signal generation and backtesting
- Performance metric calculation and visualization

This script can be used to:
1. Test model persistence functionality
2. Validate that saved models maintain prediction consistency
3. Verify continuous learning capabilities
4. Simulate real-world incremental model training
5. Evaluate trading performance after model updates

Usage:
    python strategies/LorentzianStrategy/test_lorentzian_save.py

The script will load test data, train models, save/load states, and output
performance metrics along with visualization plots.
"""

# TEST TOOL: Lorentzian Model Save/Load Functionality

This script tests the saving and loading functionality of the Lorentzian ANN model.
It's used to verify that model persistence works correctly and that model state can be
properly saved and restored for continuous learning.

Key features:
// ... existing code ... 