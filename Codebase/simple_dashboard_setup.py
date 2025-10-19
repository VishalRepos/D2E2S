#!/usr/bin/env python3
"""
Simple setup to view both 14res and 15res results in dashboard
"""

import json
import os

def create_dashboard_instructions():
    """Create instructions for viewing both studies"""
    
    print("🚀 Dashboard Setup for Both 14res and 15res Results")
    print("=" * 60)
    
    # Check existing databases
    results_dir = "optuna_results"
    
    print("📊 Available Study Databases:")
    
    if os.path.exists(f"{results_dir}/balanced_optimization.db"):
        print("   ✅ balanced_optimization.db (14res dataset)")
        print("      - Study: d2e2s_14res_balanced_1759896139")
        print("      - 25 trials completed")
        
    if os.path.exists(f"{results_dir}/15res_optimization.db"):
        print("   ✅ 15res_optimization.db (15res dataset)")
        print("      - Study: d2e2s_15res_balanced_1760675899")
        print("      - 5 trials completed")
    
    # Load and display results summary
    print("\n📈 Results Summary:")
    
    # 14res results
    try:
        with open(f'{results_dir}/d2e2s_14res_balanced_1759896139_best_params.json', 'r') as f:
            best_14res = json.load(f)
        print(f"   🏆 14res Best F1: {best_14res['best_value']:.4f}")
        print(f"      - GCN Type: {best_14res['best_params']['gcn_type']}")
        print(f"      - Batch Size: {best_14res['best_params']['batch_size']}")
        print(f"      - Learning Rate: {best_14res['best_params']['lr']:.2e}")
    except:
        print("   ⚠️  14res results not found")
    
    # 15res results  
    try:
        with open(f'{results_dir}/d2e2s_15res_balanced_1760675899_best_params.json', 'r') as f:
            best_15res = json.load(f)
        print(f"   🏆 15res Best F1: {best_15res['best_value']:.4f}")
        print(f"      - GCN Type: {best_15res['best_params']['gcn_type']}")
        print(f"      - Batch Size: {best_15res['best_params']['batch_size']}")
        print(f"      - Learning Rate: {best_15res['best_params']['lr']:.2e}")
    except:
        print("   ⚠️  15res results not found")
    
    print("\n🌐 Dashboard Access Options:")
    print("=" * 60)
    
    print("Option 1: View 14res Results")
    print("   export PATH=\"$PATH:/Users/vishal.thenuwara/.local/bin\"")
    print("   optuna-dashboard sqlite:///optuna_results/balanced_optimization.db --port 8080")
    print("   Open: http://localhost:8080")
    
    print("\nOption 2: View 15res Results (JSON-based)")
    print("   # 15res results are available in JSON format:")
    print("   # - d2e2s_15res_balanced_1760675899_best_params.json")
    print("   # - d2e2s_15res_balanced_1760675899_all_trials.json")
    print("   # - d2e2s_15res_balanced_1760675899_stats.json")
    
    print("\nOption 3: Compare Both Results")
    print("   # Use the JSON files to compare:")
    print("   cat optuna_results/d2e2s_14res_balanced_1759896139_best_params.json")
    print("   cat optuna_results/d2e2s_15res_balanced_1760675899_best_params.json")
    
    print("\n📋 Quick Comparison:")
    print("=" * 60)
    
    try:
        with open(f'{results_dir}/d2e2s_14res_balanced_1759896139_best_params.json', 'r') as f:
            best_14res = json.load(f)
        with open(f'{results_dir}/d2e2s_15res_balanced_1760675899_best_params.json', 'r') as f:
            best_15res = json.load(f)
            
        print("| Parameter | 14res | 15res |")
        print("|-----------|-------|-------|")
        print(f"| F1 Score | {best_14res['best_value']:.4f} | {best_15res['best_value']:.4f} |")
        print(f"| GCN Type | {best_14res['best_params']['gcn_type']} | {best_15res['best_params']['gcn_type']} |")
        print(f"| Batch Size | {best_14res['best_params']['batch_size']} | {best_15res['best_params']['batch_size']} |")
        print(f"| Learning Rate | {best_14res['best_params']['lr']:.2e} | {best_15res['best_params']['lr']:.2e} |")
        print(f"| GCN Layers | {best_14res['best_params']['gcn_layers']} | {best_15res['best_params']['gcn_layers']} |")
        print(f"| Attention Heads | {best_14res['best_params']['attention_heads']} | {best_15res['best_params']['attention_heads']} |")
        print(f"| Epochs | {best_14res['best_params']['epochs']} | {best_15res['best_params']['epochs']} |")
        
    except Exception as e:
        print(f"   ⚠️  Could not load comparison data: {e}")

def main():
    create_dashboard_instructions()

if __name__ == "__main__":
    main()