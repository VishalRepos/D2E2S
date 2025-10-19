#!/usr/bin/env python3
"""
Simple Optuna Results Viewer
"""

import json
import os
from pathlib import Path

def view_latest_results():
    """View the latest Optuna optimization results"""
    
    results_dir = Path("optuna_results")
    if not results_dir.exists():
        print("❌ No optuna_results directory found")
        return
    
    # Find the latest best_params file
    best_params_files = list(results_dir.glob("*_best_params.json"))
    if not best_params_files:
        print("❌ No results files found")
        return
    
    # Get the most recent file
    latest_file = max(best_params_files, key=lambda x: x.stat().st_mtime)
    
    print(f"📊 Latest Optuna Results: {latest_file.name}")
    print("="*60)
    
    # Load and display results
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    print(f"🎯 Study: {results['study_name']}")
    print(f"📊 Dataset: {results['dataset']}")
    print(f"🎲 Total Trials: {results['n_trials']}")
    print(f"🏆 Best Score: {results['best_value']:.4f}")
    
    print(f"\n🏆 Best Parameters:")
    for key, value in results['best_params'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2e}")
        else:
            print(f"  {key}: {value}")
    
    # Also show stats if available
    stats_file = latest_file.parent / latest_file.name.replace("_best_params.json", "_stats.json")
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        
        print(f"\n📈 Optimization Statistics:")
        print(f"  ✅ Completed Trials: {stats['completed_trials']}")
        print(f"  ✂️  Pruned Trials: {stats['pruned_trials']}")
        print(f"  ❌ Failed Trials: {stats['failed_trials']}")
        print(f"  📊 Success Rate: {stats['success_rate']*100:.1f}%")
        print(f"  ⚡ Pruning Rate: {stats['pruning_rate']*100:.1f}%")
    
    print("="*60)

if __name__ == "__main__":
    view_latest_results()