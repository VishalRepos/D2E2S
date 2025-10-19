#!/usr/bin/env python3
"""
Optuna Hyperparameter Tuner for D2E2S Model
Advanced hyperparameter optimization using Optuna framework
"""

import optuna
import argparse
import os
import sys
import json
import time
import subprocess
import tempfile
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

class OptunaD2E2STuner:
    def __init__(self, dataset="14res", n_trials=50, study_name=None, storage=None):
        self.dataset = dataset
        self.n_trials = n_trials
        self.study_name = study_name or f"d2e2s_{dataset}_{int(time.time())}"
        self.storage = storage
        self.results_dir = Path("optuna_results")
        self.results_dir.mkdir(exist_ok=True)
        
    def objective(self, trial):
        """Optuna objective function"""
        
        # Suggest hyperparameters
        params = {
            # Training parameters
            'batch_size': trial.suggest_categorical('batch_size', [4, 8, 16, 32]),
            'lr': trial.suggest_float('lr', 1e-6, 1e-3, log=True),
            'lr_warmup': trial.suggest_float('lr_warmup', 0.05, 0.3),
            'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True),
            'epochs': trial.suggest_int('epochs', 20, 120),
            
            # Model architecture
            'hidden_dim': trial.suggest_categorical('hidden_dim', [384, 512, 768, 1024]),
            'gcn_dim': trial.suggest_categorical('gcn_dim', [300, 512, 768, 1024]),
            'gcn_layers': trial.suggest_int('gcn_layers', 2, 4),
            'attention_heads': trial.suggest_categorical('attention_heads', [4, 8, 12, 16]),
            
            # GCN configuration
            'gcn_type': trial.suggest_categorical('gcn_type', [
                'improved', 'adaptive', 'gatv2', 'gcn', 'sage', 'hybrid'
            ]),
            
            # Regularization
            'drop_out_rate': trial.suggest_float('drop_out_rate', 0.1, 0.5),
            'gcn_dropout': trial.suggest_float('gcn_dropout', 0.05, 0.3),
            'prop_drop': trial.suggest_float('prop_drop', 0.01, 0.2),
            
            # Memory optimization
            'max_span_size': trial.suggest_int('max_span_size', 4, 10),
            'max_pairs': trial.suggest_categorical('max_pairs', [500, 800, 1000, 1500]),
            'neg_entity_count': trial.suggest_int('neg_entity_count', 30, 100),
            'neg_triple_count': trial.suggest_int('neg_triple_count', 30, 100),
            
            # Advanced features
            'use_residual': trial.suggest_categorical('use_residual', [True, False]),
            'use_layer_norm': trial.suggest_categorical('use_layer_norm', [True, False]),
            'use_multi_scale': trial.suggest_categorical('use_multi_scale', [True, False]),
            'use_graph_attention': trial.suggest_categorical('use_graph_attention', [True, False]),
            
            # Other parameters
            'sen_filter_threshold': trial.suggest_float('sen_filter_threshold', 0.3, 0.7),
        }
        
        try:
            # Run training with suggested parameters
            score = self._run_training_trial(trial.number, params)
            
            # Log intermediate results
            trial.set_user_attr('score', score)
            trial.set_user_attr('params', params)
            
            return score
            
        except Exception as e:
            print(f"Trial {trial.number} failed: {str(e)}")
            # Return a very low score for failed trials
            return -1.0
    
    def _run_training_trial(self, trial_number, params):
        """Run a single training trial"""
        
        # Create temporary parameter file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            param_file = f.name
            self._write_parameter_file(f, params)
        
        try:
            # Create trial log directory
            trial_log_dir = self.results_dir / f"trial_{trial_number}"
            trial_log_dir.mkdir(exist_ok=True)
            
            # Run training
            cmd = [
                sys.executable, "train_improved.py",
                "--param_file", param_file,
                "--trial_log_dir", str(trial_log_dir)
            ]
            
            print(f"🚀 Running trial {trial_number} with parameters:")
            print(f"   Batch size: {params['batch_size']}, LR: {params['lr']:.2e}")
            print(f"   GCN type: {params['gcn_type']}, Layers: {params['gcn_layers']}")
            print(f"   Attention heads: {params['attention_heads']}")
            
            # Run with timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
                cwd=os.getcwd()
            )
            
            if result.returncode != 0:
                print(f"❌ Trial {trial_number} failed with return code: {result.returncode}")
                print(f"Error output: {result.stderr}")
                return -1.0
            
            # Extract F1 score from output
            score = self._extract_f1_score(result.stdout)
            print(f"✅ Trial {trial_number} completed with F1 score: {score:.4f}")
            
            return score
            
        except subprocess.TimeoutExpired:
            print(f"⏰ Trial {trial_number} timed out")
            return -1.0
        except Exception as e:
            print(f"❌ Trial {trial_number} error: {str(e)}")
            return -1.0
        finally:
            # Clean up parameter file
            if os.path.exists(param_file):
                os.unlink(param_file)
    
    def _write_parameter_file(self, file_handle, params):
        """Write parameter file for training"""
        
        content = f'''# Auto-generated parameters for Optuna trial
import argparse
import torch

def train_argparser_improved():
    dataset_files = {{
        "14res": {{
            "train": "./data/14res/train_dep_triple_polarity_result.json",
            "test": "./data/14res/test_dep_triple_polarity_result.json",
            "types_path": "./data/types.json",
        }},
        "15res": {{
            "train": "./data/15res/train_dep_triple_polarity_result.json",
            "test": "./data/15res/test_dep_triple_polarity_result.json",
            "types_path": "./data/types.json",
        }},
        "16res": {{
            "train": "./data/16res/train_dep_triple_polarity_result.json",
            "test": "./data/16res/test_dep_triple_polarity_result.json",
            "types_path": "./data/types.json",
        }},
        "14lap": {{
            "train": "./data/14lap/train_dep_triple_polarity_result.json",
            "test": "./data/14lap/test_dep_triple_polarity_result.json",
            "types_path": "./data/types.json",
        }},
    }}

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="{self.dataset}", type=str)
    
    # Optuna-suggested parameters
    parser.add_argument("--batch_size", type=int, default={params['batch_size']})
    parser.add_argument("--lr", type=float, default={params['lr']})
    parser.add_argument("--lr_warmup", type=float, default={params['lr_warmup']})
    parser.add_argument("--weight_decay", type=float, default={params['weight_decay']})
    parser.add_argument("--epochs", type=int, default={params['epochs']})
    parser.add_argument("--hidden_dim", type=int, default={params['hidden_dim']})
    parser.add_argument("--gcn_dim", type=int, default={params['gcn_dim']})
    parser.add_argument("--gcn_layers", type=int, default={params['gcn_layers']})
    parser.add_argument("--attention_heads", type=int, default={params['attention_heads']})
    parser.add_argument("--gcn_type", default="{params['gcn_type']}")
    parser.add_argument("--drop_out_rate", type=float, default={params['drop_out_rate']})
    parser.add_argument("--gcn_dropout", type=float, default={params['gcn_dropout']})
    parser.add_argument("--prop_drop", type=float, default={params['prop_drop']})
    parser.add_argument("--max_span_size", type=int, default={params['max_span_size']})
    parser.add_argument("--max_pairs", type=int, default={params['max_pairs']})
    parser.add_argument("--neg_entity_count", type=int, default={params['neg_entity_count']})
    parser.add_argument("--neg_triple_count", type=int, default={params['neg_triple_count']})
    parser.add_argument("--use_residual", default={params['use_residual']})
    parser.add_argument("--use_layer_norm", default={params['use_layer_norm']})
    parser.add_argument("--use_multi_scale", default={params['use_multi_scale']})
    parser.add_argument("--use_graph_attention", default={params['use_graph_attention']})
    parser.add_argument("--sen_filter_threshold", type=float, default={params['sen_filter_threshold']})
    
    # Fixed parameters
    parser.add_argument("--use_improved_gcn", default=True)
    parser.add_argument("--emb_dim", type=int, default=1536)
    parser.add_argument("--deberta_feature_dim", type=int, default=1536)
    parser.add_argument("--pretrained_deberta_name", default="microsoft/deberta-v2-xxlarge")
    parser.add_argument("--is_bidirect", default=True)
    parser.add_argument("--lstm_layers", type=int, default=2)
    parser.add_argument("--mem_dim", type=int, default=768)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_path", type=str, default="log/")
    parser.add_argument("--store_predictions", action="store_true", default=True)
    parser.add_argument("--store_examples", action="store_true", default=True)
    
    opt = parser.parse_args([])
    opt.label = opt.dataset
    opt.dataset_file = dataset_files[opt.dataset]
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return opt
'''
        file_handle.write(content)
    
    def _extract_f1_score(self, output):
        """Extract F1 score from training output"""
        
        import re
        
        # Look for the best F1 score pattern
        patterns = [
            r"Best F1 score:\s*([0-9]*\.?[0-9]+)",
            r"🏆 Best F1 score:\s*([0-9]*\.?[0-9]+)",
            r"mic_f1_score['\"]:\s*([0-9]*\.?[0-9]+)"
        ]
        
        best_score = -1.0
        
        for pattern in patterns:
            matches = re.findall(pattern, output)
            if matches:
                try:
                    score = float(matches[-1])  # Take the last (best) score
                    if score > best_score:
                        best_score = score
                except ValueError:
                    continue
        
        return best_score
    
    def run_optimization(self):
        """Run Optuna optimization"""
        
        print(f"🚀 Starting Optuna hyperparameter optimization")
        print(f"📊 Dataset: {self.dataset}")
        print(f"🎯 Number of trials: {self.n_trials}")
        print(f"📝 Study name: {self.study_name}")
        print("="*60)
        
        # Create or load study
        if self.storage:
            study = optuna.create_study(
                direction='maximize',
                study_name=self.study_name,
                storage=self.storage,
                load_if_exists=True
            )
        else:
            study = optuna.create_study(direction='maximize')
        
        # Add custom callbacks
        def print_callback(study, trial):
            print(f"🎯 Trial {trial.number}: F1 = {trial.value:.4f}")
            if study.best_trial.number == trial.number:
                print(f"🏆 New best score: {trial.value:.4f}")
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            callbacks=[print_callback],
            show_progress_bar=True
        )
        
        # Save results
        self._save_results(study)
        
        # Print final results
        self._print_results(study)
        
        return study
    
    def _save_results(self, study):
        """Save optimization results"""
        
        # Save study object
        study_file = self.results_dir / f"{self.study_name}_study.pkl"
        with open(study_file, 'wb') as f:
            import pickle
            pickle.dump(study, f)
        
        # Save best parameters
        best_params_file = self.results_dir / f"{self.study_name}_best_params.json"
        with open(best_params_file, 'w') as f:
            json.dump({
                'best_value': study.best_value,
                'best_params': study.best_params,
                'n_trials': len(study.trials),
                'study_name': self.study_name,
                'dataset': self.dataset
            }, f, indent=2)
        
        # Save all trials
        trials_file = self.results_dir / f"{self.study_name}_all_trials.json"
        trials_data = []
        for trial in study.trials:
            trials_data.append({
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name
            })
        
        with open(trials_file, 'w') as f:
            json.dump(trials_data, f, indent=2)
    
    def _print_results(self, study):
        """Print optimization results"""
        
        print("\n" + "="*60)
        print("🎉 OPTUNA OPTIMIZATION COMPLETED! 🎉")
        print("="*60)
        print(f"📊 Dataset: {self.dataset}")
        print(f"🎯 Total trials: {len(study.trials)}")
        print(f"🏆 Best F1 score: {study.best_value:.4f}")
        print(f"✅ Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
        
        print(f"\n🏆 Best parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        
        print(f"\n📁 Results saved to: {self.results_dir}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning for D2E2S")
    parser.add_argument("--dataset", default="14res", choices=["14res", "15res", "16res", "14lap"])
    parser.add_argument("--n_trials", type=int, default=50, help="Number of trials")
    parser.add_argument("--study_name", type=str, help="Study name for Optuna")
    parser.add_argument("--storage", type=str, help="Database URL for study storage")
    parser.add_argument("--dashboard", action="store_true", help="Launch Optuna dashboard")
    
    args = parser.parse_args()
    
    if args.dashboard and args.storage:
        print(f"🌐 Launching Optuna dashboard...")
        print(f"📊 Access at: http://localhost:8080")
        os.system(f"optuna-dashboard {args.storage}")
        return
    
    # Create tuner and run optimization
    tuner = OptunaD2E2STuner(
        dataset=args.dataset,
        n_trials=args.n_trials,
        study_name=args.study_name,
        storage=args.storage
    )
    
    study = tuner.run_optimization()
    
    print(f"\n🎯 To visualize results, run:")
    print(f"optuna-dashboard sqlite:///optuna_results/{tuner.study_name}.db")


if __name__ == "__main__":
    main()