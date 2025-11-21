#!/usr/bin/env python3
"""
Kaggle Training Script for D2E2S
Usage: python kaggle_train.py --dataset 15res --epochs 100
"""

import argparse
import subprocess
import sys
import os

def run_training(dataset, epochs, batch_size=8, lr=0.0001716):
    """Run D2E2S training with optimal parameters"""
    
    # Change to codebase directory
    os.chdir("Codebase")
    
    # Training command with verified working parameters
    cmd = [
        sys.executable, "train.py",
        "--dataset", dataset,
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--epochs", str(epochs),
        "--gcn_type", "adaptive",
        "--attention_heads", "12",
        "--lr_warmup", "0",
        "--weight_decay", "0"
    ]
    
    # Add GPU/CPU selection
    if os.system("nvidia-smi") == 0:
        print("GPU detected, using CUDA")
        cmd.extend(["--device", "cuda"])
    else:
        print("No GPU detected, using CPU")
        cmd.extend(["--device", "cpu"])
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description="D2E2S Kaggle Training")
    parser.add_argument("--dataset", default="15res", choices=["14res", "15res", "16res", "14lap"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0001716)
    
    args = parser.parse_args()
    
    print(f"Starting D2E2S training: {args.dataset} for {args.epochs} epochs")
    run_training(args.dataset, args.epochs, args.batch_size, args.lr)

if __name__ == "__main__":
    main()