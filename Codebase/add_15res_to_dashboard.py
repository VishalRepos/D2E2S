#!/usr/bin/env python3
"""
Add 15res results to the existing Optuna database for dashboard visualization
"""

import optuna
import json
import time

def add_15res_study():
    """Add 15res study to the balanced_optimization.db"""
    
    # Connect to existing database
    storage = optuna.storages.RDBStorage(
        url="sqlite:///optuna_results/balanced_optimization.db"
    )
    
    # Create new study for 15res
    study_name = "d2e2s_15res_balanced_1760675899"
    
    try:
        # Try to load existing study first
        study = optuna.load_study(
            study_name=study_name,
            storage=storage
        )
        print(f"✅ Study {study_name} already exists")
    except:
        # Create new study
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            storage=storage
        )
        print(f"✅ Created new study: {study_name}")
    
    # Load 15res trial results
    with open('optuna_results/d2e2s_15res_balanced_1760675899_all_trials.json', 'r') as f:
        trials_data = json.load(f)
    
    print(f"📊 Adding {len(trials_data)} trials to study...")
    
    # Add trials to study
    for trial_data in trials_data:
        # Create trial
        trial = study.ask()
        
        # Set parameters
        for param_name, param_value in trial_data['params'].items():
            trial.suggest_categorical(param_name, [param_value])
        
        # Report result
        study.tell(trial, trial_data['value'])
        
        print(f"   Trial {trial_data['number']}: F1 = {trial_data['value']:.4f}")
    
    print(f"✅ Successfully added 15res study to database")
    print(f"🏆 Best value: {study.best_value:.4f}")
    
    return study

def main():
    print("🚀 Adding 15res results to Optuna dashboard database...")
    print("=" * 60)
    
    study = add_15res_study()
    
    print("\n🌐 Dashboard now contains both studies:")
    print("   - d2e2s_14res_balanced_1759896139 (14res dataset)")
    print("   - d2e2s_15res_balanced_1760675899 (15res dataset)")
    print("\n📊 Refresh your dashboard to see both studies!")
    print("🔗 Dashboard URL: http://localhost:8080")

if __name__ == "__main__":
    main()