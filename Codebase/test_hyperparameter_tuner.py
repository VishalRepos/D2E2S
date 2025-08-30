#!/usr/bin/env python3
"""
Test script for the hyperparameter tuner to verify real-time output display
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(os.getcwd())

from simple_hyperparameter_tuner import SimpleHyperparameterTuner

def test_hyperparameter_tuner():
    """Test the hyperparameter tuner with a small number of trials"""
    
    print("🧪 Testing Hyperparameter Tuner with Real-time Output")
    print("="*60)
    
    # Create tuner with just 2 trials for testing
    tuner = SimpleHyperparameterTuner(
        dataset="14res",
        n_trials=2,
        verbose=True
    )
    
    # Preview the combinations
    print("\n📋 Previewing parameter combinations:")
    tuner.preview_combinations()
    
    # Ask user if they want to proceed
    response = input("\n❓ Do you want to run the optimization? (y/n): ")
    if response.lower() != 'y':
        print("❌ Test cancelled by user")
        return
    
    # Run optimization
    print("\n🚀 Starting optimization test...")
    tuner.run_optimization()
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    test_hyperparameter_tuner()
