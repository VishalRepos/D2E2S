#!/usr/bin/env python3
"""
Kaggle Setup Script for D2E2S Training
Usage: python kaggle_setup.py --dataset 15res --epochs 100
"""

import subprocess
import sys
import os

def install_dependencies():
    """Install required packages for Kaggle environment"""
    packages = [
        "torch==2.4.0",
        "transformers==4.28.1", 
        "torch-geometric==2.3.1",
        "sentencepiece",
        "optuna>=3.0.0",
        "pandas>=1.3.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0"
    ]
    
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def setup_environment():
    """Set up Kaggle environment variables"""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    print("Setting up D2E2S environment for Kaggle...")
    install_dependencies()
    setup_environment()
    print("Setup complete! Ready for training.")

if __name__ == "__main__":
    main()