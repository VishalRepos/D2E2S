#!/usr/bin/env python3
"""
Train D2E2S model with optimal hyperparameters and get actual F1, Precision, Recall scores
"""

import subprocess
import sys
import json
from pathlib import Path

def get_optimal_params():
    """Get optimal hyperparameters for each dataset"""
    
    optimal_configs = {
        "15res": {
            "name": "15res - CHAMPION",
            "expected_f1": "86.44%",
            "command": [
                "python3", "train.py",
                "--dataset", "15res",
                "--batch_size", "6",
                "--lr", "0.000312",
                "--epochs", "40",
                "--gcn_type", "hybrid",
                "--attention_heads", "16",
                "--hidden_dim", "1024",
                "--gcn_dim", "768",
                "--lr_warmup", "0.12",
                "--weight_decay", "0.0012"
            ]
        },
        "16res": {
            "name": "16res - RUNNER-UP",
            "expected_f1": "86.41%",
            "command": [
                "python3", "train.py",
                "--dataset", "16res",
                "--batch_size", "6",
                "--lr", "0.000189",
                "--epochs", "40",
                "--gcn_type", "hybrid",
                "--attention_heads", "10",
                "--hidden_dim", "768",
                "--gcn_dim", "512",
                "--lr_warmup", "0.08",
                "--weight_decay", "0.0008"
            ]
        },
        "14res": {
            "name": "14res - CORRECTED",
            "expected_f1": "84.23%",
            "command": [
                "python3", "train.py",
                "--dataset", "14res",
                "--batch_size", "6",
                "--lr", "0.000189",
                "--epochs", "40",
                "--gcn_type", "hybrid",
                "--attention_heads", "10",
                "--hidden_dim", "768",
                "--gcn_dim", "512",
                "--lr_warmup", "0.08",
                "--weight_decay", "0.0008"
            ]
        },
        "14lap": {
            "name": "14lap - IMPROVED",
            "expected_f1": "82.94%",
            "command": [
                "python3", "train.py",
                "--dataset", "14lap",
                "--batch_size", "6",
                "--lr", "0.000156",
                "--epochs", "40",
                "--gcn_type", "hybrid",
                "--attention_heads", "16",
                "--hidden_dim", "1024",
                "--gcn_dim", "768",
                "--lr_warmup", "0.12",
                "--weight_decay", "0.0012"
            ]
        }
    }
    
    return optimal_configs

def train_model(dataset, config):
    """Train model with optimal parameters"""
    
    print(f"🚀 Training {config['name']}")
    print(f"📊 Expected F1: {config['expected_f1']}")
    print(f"⚙️  Command: {' '.join(config['command'])}")
    print("=" * 60)
    
    try:
        # Run training
        result = subprocess.run(
            config['command'],
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout
            cwd="."
        )
        
        if result.returncode == 0:
            print(f"✅ {dataset} training completed successfully!")
            print("📊 Training output:")
            print(result.stdout[-1000:])  # Last 1000 characters
            return True, result.stdout
        else:
            print(f"❌ {dataset} training failed!")
            print("Error:", result.stderr)
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {dataset} training timed out (2 hours)")
        return False, "Timeout"
    except Exception as e:
        print(f"❌ {dataset} training error: {str(e)}")
        return False, str(e)

def main():
    print("🎯 D2E2S Model Training with Optimal Hyperparameters")
    print("=" * 60)
    print("📋 This will train models with your optimized hyperparameters")
    print("📊 and provide actual F1, Precision, Recall scores")
    print()
    
    # Get optimal configurations
    configs = get_optimal_params()
    
    # Ask user which dataset to train
    print("📊 Available datasets:")
    for i, (dataset, config) in enumerate(configs.items(), 1):
        print(f"   {i}. {config['name']} (Expected: {config['expected_f1']})")
    
    print("   5. Train all datasets")
    print()
    
    choice = input("🎯 Select dataset to train (1-5): ").strip()
    
    if choice == "5":
        # Train all datasets
        print("🚀 Training all datasets with optimal hyperparameters...")
        results = {}
        
        for dataset, config in configs.items():
            print(f"\n{'='*60}")
            success, output = train_model(dataset, config)
            results[dataset] = {"success": success, "output": output}
            print(f"{'='*60}\n")
        
        # Summary
        print("📊 TRAINING SUMMARY:")
        for dataset, result in results.items():
            status = "✅ Success" if result["success"] else "❌ Failed"
            print(f"   {dataset}: {status}")
            
    else:
        # Train single dataset
        dataset_list = list(configs.keys())
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(dataset_list):
                dataset = dataset_list[idx]
                config = configs[dataset]
                success, output = train_model(dataset, config)
                
                if success:
                    print(f"\n🎉 {dataset} training completed!")
                    print("📊 Check the output above for F1, Precision, Recall scores")
                else:
                    print(f"\n❌ {dataset} training failed")
            else:
                print("❌ Invalid choice")
        except ValueError:
            print("❌ Invalid input")

if __name__ == "__main__":
    main()