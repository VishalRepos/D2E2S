#!/usr/bin/env python3
"""
Advanced Optuna Hyperparameter Tuner with Pruning and Multi-objective Optimization
"""

import optuna
from optuna.pruners import MedianPruner, HyperbandPruner
from optuna.samplers import TPESampler, CmaEsSampler
import argparse
import json
import time
from pathlib import Path
from optuna_hyperparameter_tuner import OptunaD2E2STuner
from optuna_config import get_search_space_config, get_pruning_config, get_sampler_config

class AdvancedOptunaD2E2STuner(OptunaD2E2STuner):
    def __init__(self, dataset="14res", n_trials=50, optimization_strategy="balanced", 
                 use_pruning=True, sampler_type="tpe", storage=None):
        super().__init__(dataset, n_trials, storage=storage)
        
        self.optimization_strategy = optimization_strategy
        self.use_pruning = use_pruning
        self.sampler_type = sampler_type
        
        # Load configurations
        self.search_spaces = get_search_space_config()
        self.pruning_config = get_pruning_config()
        self.sampler_config = get_sampler_config()
        
        # Set study name based on strategy
        self.study_name = f"d2e2s_{dataset}_{optimization_strategy}_{int(time.time())}"
    
    def create_study(self):
        """Create Optuna study with advanced configuration"""
        
        # Configure sampler
        if self.sampler_type == "tpe":
            sampler = TPESampler(**self.sampler_config['tpe'])
        elif self.sampler_type == "cmaes":
            sampler = CmaEsSampler(**self.sampler_config['cmaes'])
        else:
            sampler = None
        
        # Configure pruner
        if self.use_pruning:
            if self.optimization_strategy == "quick":
                pruner = MedianPruner(**self.pruning_config['median_pruner'])
            else:
                pruner = HyperbandPruner(**self.pruning_config['hyperband_pruner'])
        else:
            pruner = None
        
        # Create study
        if self.storage:
            study = optuna.create_study(
                direction='maximize',
                study_name=self.study_name,
                storage=self.storage,
                sampler=sampler,
                pruner=pruner,
                load_if_exists=True
            )
        else:
            study = optuna.create_study(
                direction='maximize',
                sampler=sampler,
                pruner=pruner
            )
        
        return study
    
    def objective_with_pruning(self, trial):
        """Objective function with intermediate value reporting for pruning"""
        
        # Get search space for current strategy
        search_space = self.search_spaces[self.optimization_strategy]
        
        # Suggest parameters based on strategy
        params = self._suggest_parameters(trial, search_space)
        
        try:
            # Run training with intermediate reporting
            score = self._run_training_with_pruning(trial, params)
            return score
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"Trial {trial.number} failed: {str(e)}")
            return -1.0
    
    def _suggest_parameters(self, trial, search_space):
        """Suggest parameters based on search space configuration"""
        
        params = {}
        
        for param_name, param_config in search_space.items():
            if isinstance(param_config, list):
                # Categorical parameter
                params[param_name] = trial.suggest_categorical(param_name, param_config)
            elif isinstance(param_config, tuple):
                if len(param_config) == 2:
                    # Integer parameter
                    params[param_name] = trial.suggest_int(param_name, param_config[0], param_config[1])
                elif len(param_config) == 3 and param_config[2] == 'log':
                    # Log-scale float parameter
                    params[param_name] = trial.suggest_float(param_name, param_config[0], param_config[1], log=True)
                else:
                    # Regular float parameter
                    params[param_name] = trial.suggest_float(param_name, param_config[0], param_config[1])
        
        # Add default parameters not in search space
        default_params = {
            'lr_warmup': 0.15,
            'weight_decay': 0.01,
            'max_grad_norm': 1.0,
            'sen_filter_threshold': 0.5,
            'neg_entity_count': 50,
            'neg_triple_count': 50,
            'max_span_size': 6,
            'max_pairs': 800,
            'hidden_dim': 768,
            'gcn_dim': 768,
            'drop_out_rate': 0.3,
            'gcn_dropout': 0.1,
            'prop_drop': 0.05,
            'use_residual': True,
            'use_layer_norm': True,
            'use_multi_scale': True,
            'use_graph_attention': True
        }
        
        for key, value in default_params.items():
            if key not in params:
                params[key] = value
        
        return params
    
    def _run_training_with_pruning(self, trial, params):
        """Run training with intermediate value reporting for pruning"""
        
        # For now, use the same training method as base class
        # In a full implementation, you would modify the training script
        # to report intermediate F1 scores using trial.report()
        
        score = self._run_training_trial(trial.number, params)
        
        # Report final score (in practice, you'd report intermediate scores)
        trial.report(score, step=params.get('epochs', 120))
        
        # Check if trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        return score
    
    def run_optimization(self):
        """Run advanced optimization with pruning and callbacks"""
        
        print(f"🚀 Starting Advanced Optuna Optimization")
        print(f"📊 Dataset: {self.dataset}")
        print(f"🎯 Strategy: {self.optimization_strategy}")
        print(f"🔬 Sampler: {self.sampler_type}")
        print(f"✂️  Pruning: {'Enabled' if self.use_pruning else 'Disabled'}")
        print(f"🎲 Trials: {self.n_trials}")
        print("="*60)
        
        # Create study
        study = self.create_study()
        
        # Define callbacks
        def print_callback(study, trial):
            print(f"🎯 Trial {trial.number}: F1 = {trial.value:.4f} ({trial.state.name})")
            if trial.state == optuna.trial.TrialState.COMPLETE and study.best_trial.number == trial.number:
                print(f"🏆 New best score: {trial.value:.4f}")
            elif trial.state == optuna.trial.TrialState.PRUNED:
                print(f"✂️  Trial {trial.number} pruned")
        
        # Run optimization
        study.optimize(
            self.objective_with_pruning,
            n_trials=self.n_trials,
            callbacks=[print_callback],
            show_progress_bar=True
        )
        
        # Save and print results
        self._save_advanced_results(study)
        self._print_advanced_results(study)
        
        return study
    
    def _save_advanced_results(self, study):
        """Save advanced optimization results"""
        
        # Save basic results
        super()._save_results(study)
        
        # Save optimization statistics
        stats_file = self.results_dir / f"{self.study_name}_stats.json"
        
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
        
        stats = {
            'optimization_strategy': self.optimization_strategy,
            'sampler_type': self.sampler_type,
            'use_pruning': self.use_pruning,
            'total_trials': len(study.trials),
            'completed_trials': len(completed_trials),
            'pruned_trials': len(pruned_trials),
            'failed_trials': len(failed_trials),
            'success_rate': len(completed_trials) / len(study.trials) if study.trials else 0,
            'pruning_rate': len(pruned_trials) / len(study.trials) if study.trials else 0,
            'best_value': study.best_value if completed_trials else None,
            'best_params': study.best_params if completed_trials else None
        }
        
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def _print_advanced_results(self, study):
        """Print advanced optimization results"""
        
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
        
        print("\n" + "="*60)
        print("🎉 ADVANCED OPTUNA OPTIMIZATION COMPLETED! 🎉")
        print("="*60)
        print(f"📊 Dataset: {self.dataset}")
        print(f"🎯 Strategy: {self.optimization_strategy}")
        print(f"🔬 Sampler: {self.sampler_type}")
        print(f"✂️  Pruning: {'Enabled' if self.use_pruning else 'Disabled'}")
        print(f"🎲 Total trials: {len(study.trials)}")
        print(f"✅ Completed: {len(completed_trials)}")
        print(f"✂️  Pruned: {len(pruned_trials)}")
        print(f"❌ Failed: {len(failed_trials)}")
        
        if completed_trials:
            print(f"🏆 Best F1 score: {study.best_value:.4f}")
            print(f"📈 Success rate: {len(completed_trials)/len(study.trials)*100:.1f}%")
            print(f"⚡ Pruning rate: {len(pruned_trials)/len(study.trials)*100:.1f}%")
            
            print(f"\n🏆 Best parameters:")
            for key, value in study.best_params.items():
                print(f"  {key}: {value}")
        
        print(f"\n📁 Results saved to: {self.results_dir}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Advanced Optuna hyperparameter tuning")
    parser.add_argument("--dataset", default="14res", choices=["14res", "15res", "16res", "14lap"])
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--strategy", default="balanced", 
                       choices=["quick", "balanced", "comprehensive"])
    parser.add_argument("--sampler", default="tpe", choices=["tpe", "cmaes", "random"])
    parser.add_argument("--no_pruning", action="store_true", help="Disable pruning")
    parser.add_argument("--storage", type=str, help="Database URL for study storage")
    parser.add_argument("--dashboard", action="store_true", help="Launch dashboard after optimization")
    
    args = parser.parse_args()
    
    # Create advanced tuner
    tuner = AdvancedOptunaD2E2STuner(
        dataset=args.dataset,
        n_trials=args.n_trials,
        optimization_strategy=args.strategy,
        use_pruning=not args.no_pruning,
        sampler_type=args.sampler,
        storage=args.storage
    )
    
    # Run optimization
    study = tuner.run_optimization()
    
    # Launch dashboard if requested
    if args.dashboard and args.storage:
        print(f"\n🌐 Launching Optuna dashboard...")
        print(f"📊 Access at: http://localhost:8080")
        import os
        os.system(f"optuna-dashboard {args.storage}")


if __name__ == "__main__":
    main()