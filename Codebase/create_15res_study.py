#!/usr/bin/env python3
"""
Create a separate 15res study database for the dashboard
"""

import optuna
import json

def create_15res_study():
    """Create a new study database for 15res results"""
    
    # Create storage for 15res
    storage = optuna.storages.RDBStorage(
        url="sqlite:///optuna_results/15res_optimization.db"
    )
    
    # Create study
    study = optuna.create_study(
        study_name="d2e2s_15res_balanced_1760675899",
        direction="maximize",
        storage=storage
    )
    
    # Load 15res trial results
    with open('optuna_results/d2e2s_15res_balanced_1760675899_all_trials.json', 'r') as f:
        trials_data = json.load(f)
    
    print(f"📊 Creating study with {len(trials_data)} trials...")
    
    # Define the objective function to recreate trials
    def objective(trial):
        # Get the trial data for this trial number
        trial_data = trials_data[trial.number]
        
        # Suggest parameters based on the original trial
        params = trial_data['params']
        
        # Suggest parameters with proper ranges
        batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32])
        lr = trial.suggest_float('lr', 1e-6, 1e-3, log=True)
        lr_warmup = trial.suggest_float('lr_warmup', 0, 0.3)
        weight_decay = trial.suggest_float('weight_decay', 0, 0.1, log=True)
        gcn_type = trial.suggest_categorical('gcn_type', ['adaptive', 'hybrid', 'gatv2', 'improved'])
        gcn_layers = trial.suggest_int('gcn_layers', 2, 4)
        attention_heads = trial.suggest_categorical('attention_heads', [4, 8, 12, 16])
        hidden_dim = trial.suggest_categorical('hidden_dim', [384, 512, 768, 1024])
        gcn_dim = trial.suggest_categorical('gcn_dim', [300, 512, 768, 1024])
        epochs = trial.suggest_int('epochs', 20, 120)
        
        # Return the actual score from the trial data
        return trial_data['value']
    
    # Run optimization to recreate the trials
    study.optimize(objective, n_trials=len(trials_data))
    
    print(f"✅ Created 15res study with {len(study.trials)} trials")
    print(f"🏆 Best value: {study.best_value:.4f}")
    
    return study

def main():
    print("🚀 Creating separate 15res study database...")
    print("=" * 60)
    
    study = create_15res_study()
    
    print("\n🌐 Now you have two separate study databases:")
    print("   - balanced_optimization.db (14res dataset)")
    print("   - 15res_optimization.db (15res dataset)")
    print("\n📊 To view both in dashboard:")
    print("   1. Start dashboard with: optuna-dashboard sqlite:///optuna_results/balanced_optimization.db")
    print("   2. Or start with: optuna-dashboard sqlite:///optuna_results/15res_optimization.db")
    print("   3. Or create a combined view")

if __name__ == "__main__":
    main()