#!/usr/bin/env python3
"""
Simple Hyperparameter Tuner for DeBERTa-v2-XXLarge
This script provides an easy-to-use interface for hyperparameter optimization
"""

import argparse
import os
import json
import time
import warnings
from pathlib import Path
import subprocess
import sys

warnings.filterwarnings("ignore")

class SimpleHyperparameterTuner:
    def __init__(self, dataset="14res", n_trials=20, config_file="hyperparameter_config.py", verbose=False):
        self.dataset = dataset
        self.n_trials = n_trials
        self.verbose = verbose
        self.results_dir = Path("hyperparameter_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Load configuration from Python file
        self.config = self._load_config(config_file)
        
        # Get predefined parameter combinations from config
        self.parameter_combinations = self.config.get('predefined_combinations', [])
        
        # Add random variations for remaining trials
        self._generate_random_variations()
    
    def _load_config(self, config_file):
        """Load configuration from Python file"""
        try:
            # Import the configuration module
            import importlib.util
            spec = importlib.util.spec_from_file_location("hyperparameter_config", config_file)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            
            # Get configuration using the module's function
            if hasattr(config_module, 'get_config'):
                return config_module.get_config()
            else:
                # Fallback: create config from module variables
                return {
                    'predefined_combinations': getattr(config_module, 'PREDEFINED_COMBINATIONS', []),
                    'search_spaces': getattr(config_module, 'SEARCH_SPACES', {}),
                    'optimization': {
                        'n_trials': getattr(config_module, 'N_TRIALS', 50),
                        'timeout': getattr(config_module, 'TIMEOUT', 7200)
                    }
                }
                
        except Exception as e:
            print(f"Warning: Could not load config from {config_file}: {e}")
            print("Using default configuration...")
            # Return default configuration
            return {
                'predefined_combinations': [],
                'search_spaces': {},
                'optimization': {'n_trials': 50, 'timeout': 7200}
            }
    
    def _generate_random_variations(self):
        """Generate completely random parameter combinations with no duplicates"""
        
        import random
        
        # Get search spaces from config for randomization
        search_spaces = self.config.get('search_spaces', {})
        
        # Track used combinations to avoid duplicates
        used_combinations = set()
        
        # Generate all trials randomly since no predefined combinations exist
        for i in range(self.n_trials):
            attempts = 0
            max_attempts = 100  # Prevent infinite loops
            
            while attempts < max_attempts:
                # Generate completely random parameters from scratch
                new_params = self._generate_random_params_from_scratch(search_spaces)
                new_params['description'] = f'Random trial {i+1}'
                
                # Check if this combination is unique
                combo_key = self._get_combination_key(new_params)
                
                if combo_key not in used_combinations:
                    used_combinations.add(combo_key)
                    self.parameter_combinations.append(new_params)
                    break
                
                attempts += 1
            
            if attempts >= max_attempts:
                print(f"Warning: Could not generate unique combination for trial {i+1} after {max_attempts} attempts")
                # Fallback: generate with slight modification
                fallback_params = self._generate_random_params_from_scratch(search_spaces)
                fallback_params['description'] = f'Fallback trial {i+1}'
                self.parameter_combinations.append(fallback_params)
    
    def _generate_random_params(self, search_spaces, base_params):
        """Generate random parameters using search spaces configuration"""
        
        import random
        
        new_params = base_params.copy()
        
        # Randomize categorical parameters
        if 'batch_size' in search_spaces and search_spaces['batch_size']['type'] == 'categorical':
            new_params['batch_size'] = random.choice(search_spaces['batch_size']['values'])
        
        if 'hidden_dim' in search_spaces and search_spaces['hidden_dim']['type'] == 'categorical':
            new_params['hidden_dim'] = random.choice(search_spaces['hidden_dim']['values'])
        
        if 'gcn_dim' in search_spaces and search_spaces['gcn_dim']['type'] == 'categorical':
            new_params['gcn_dim'] = random.choice(search_spaces['gcn_dim']['values'])
        
        if 'attention_heads' in search_spaces and search_spaces['attention_heads']['type'] == 'categorical':
            new_params['attention_heads'] = random.choice(search_spaces['attention_heads']['values'])
        
        if 'gcn_type' in search_spaces and search_spaces['gcn_type']['type'] == 'categorical':
            new_params['gcn_type'] = random.choice(search_spaces['gcn_type']['values'])
        
        if 'max_pairs' in search_spaces and search_spaces['max_pairs']['type'] == 'categorical':
            new_params['max_pairs'] = random.choice(search_spaces['max_pairs']['values'])
        
        # Randomize integer parameters
        if 'gcn_layers' in search_spaces and search_spaces['gcn_layers']['type'] == 'int':
            new_params['gcn_layers'] = random.randint(
                search_spaces['gcn_layers']['low'], 
                search_spaces['gcn_layers']['high']
            )
        
        if 'max_span_size' in search_spaces and search_spaces['max_span_size']['type'] == 'int':
            new_params['max_span_size'] = random.randint(
                search_spaces['max_span_size']['low'], 
                search_spaces['max_span_size']['high']
            )
        
        if 'epochs' in search_spaces and search_spaces['epochs']['type'] == 'int':
            new_params['epochs'] = random.randint(
                search_spaces['epochs']['low'], 
                search_spaces['epochs']['high']
            )
        
        # Randomize float parameters
        if 'lr' in search_spaces and search_spaces['lr']['type'] == 'float':
            if search_spaces['lr'].get('log', False):
                new_params['lr'] = random.uniform(
                    search_spaces['lr']['low'], 
                    search_spaces['lr']['high']
                )
            else:
                new_params['lr'] = random.uniform(
                    search_spaces['lr']['low'], 
                    search_spaces['lr']['high']
                )
        
        if 'lr_warmup' in search_spaces and search_spaces['lr_warmup']['type'] == 'float':
            new_params['lr_warmup'] = random.uniform(
                search_spaces['lr_warmup']['low'], 
                search_spaces['lr_warmup']['high']
            )
        
        if 'weight_decay' in search_spaces and search_spaces['weight_decay']['type'] == 'float':
            if search_spaces['weight_decay'].get('log', False):
                new_params['weight_decay'] = random.uniform(
                    search_spaces['weight_decay']['low'], 
                    search_spaces['weight_decay']['high']
                )
            else:
                new_params['weight_decay'] = random.uniform(
                    search_spaces['weight_decay']['low'], 
                    search_spaces['weight_decay']['high']
                )
        
        if 'drop_out_rate' in search_spaces and search_spaces['drop_out_rate']['type'] == 'float':
            new_params['drop_out_rate'] = random.uniform(
                search_spaces['drop_out_rate']['low'], 
                search_spaces['drop_out_rate']['high']
            )
        
        if 'gcn_dropout' in search_spaces and search_spaces['gcn_dropout']['type'] == 'float':
            new_params['gcn_dropout'] = random.uniform(
                search_spaces['gcn_dropout']['low'], 
                search_spaces['gcn_dropout']['high']
            )
        
        # Randomize boolean parameters
        if 'use_residual' in search_spaces and search_spaces['use_residual']['type'] == 'categorical':
            new_params['use_residual'] = random.choice(search_spaces['use_residual']['values'])
        
        if 'use_layer_norm' in search_spaces and search_spaces['use_layer_norm']['type'] == 'categorical':
            new_params['use_layer_norm'] = random.choice(search_spaces['use_layer_norm']['values'])
        
        if 'use_multi_scale' in search_spaces and search_spaces['use_multi_scale']['type'] == 'categorical':
            new_params['use_multi_scale'] = random.choice(search_spaces['use_multi_scale']['values'])
        
        if 'use_graph_attention' in search_spaces and search_spaces['use_graph_attention']['type'] == 'categorical':
            new_params['use_graph_attention'] = random.choice(search_spaces['use_graph_attention']['values'])
        
        return new_params
    
    def _generate_random_params_from_scratch(self, search_spaces):
        """Generate completely random parameters from scratch - only most important ones"""
        
        import random
        
        new_params = {}
        
        # Generate categorical parameters (Critical)
        if 'batch_size' in search_spaces and search_spaces['batch_size']['type'] == 'categorical':
            new_params['batch_size'] = random.choice(search_spaces['batch_size']['values'])
        
        if 'gcn_type' in search_spaces and search_spaces['gcn_type']['type'] == 'categorical':
            new_params['gcn_type'] = random.choice(search_spaces['gcn_type']['values'])
        
        if 'attention_heads' in search_spaces and search_spaces['attention_heads']['type'] == 'categorical':
            new_params['attention_heads'] = random.choice(search_spaces['attention_heads']['values'])
        
        if 'max_pairs' in search_spaces and search_spaces['max_pairs']['type'] == 'categorical':
            new_params['max_pairs'] = random.choice(search_spaces['max_pairs']['values'])
        
        # Generate integer parameters (Critical)
        if 'gcn_layers' in search_spaces and search_spaces['gcn_layers']['type'] == 'int':
            new_params['gcn_layers'] = random.randint(
                search_spaces['gcn_layers']['low'], 
                search_spaces['gcn_layers']['high']
            )
        
        if 'epochs' in search_spaces and search_spaces['epochs']['type'] == 'int':
            new_params['epochs'] = random.randint(
                search_spaces['epochs']['low'], 
                search_spaces['epochs']['high']
            )
        
        # Generate float parameters (Critical)
        if 'lr' in search_spaces and search_spaces['lr']['type'] == 'float':
            if search_spaces['lr'].get('log', False):
                new_params['lr'] = random.uniform(
                    search_spaces['lr']['low'], 
                    search_spaces['lr']['high']
                )
            else:
                new_params['lr'] = random.uniform(
                    search_spaces['lr']['low'], 
                    search_spaces['lr']['high']
                )
        
        # Set proven default values for DeBERTa-v2-XXLarge
        new_params['lr_warmup'] = 0.15  # Conservative for XXLarge
        new_params['weight_decay'] = 0.01  # Standard regularization
        new_params['max_grad_norm'] = 1.0  # Stable for XXLarge
        new_params['hidden_dim'] = 768  # Half of emb_dim for bidirectional LSTM
        new_params['gcn_dim'] = 768  # Optimized for XXLarge
        new_params['max_span_size'] = 6  # Memory efficient for XXLarge
        new_params['neg_entity_count'] = 50  # Reduced for XXLarge memory
        new_params['neg_triple_count'] = 50  # Reduced for XXLarge memory
        new_params['use_residual'] = True  # Proven for XXLarge
        new_params['use_layer_norm'] = True  # Essential for XXLarge
        new_params['use_multi_scale'] = True  # Proven for XXLarge
        new_params['use_graph_attention'] = True  # Proven for XXLarge
        new_params['drop_out_rate'] = 0.3  # Conservative for XXLarge
        new_params['gcn_dropout'] = 0.1  # Conservative for XXLarge
        new_params['prop_drop'] = 0.05  # Conservative for XXLarge
        new_params['sen_filter_threshold'] = 0.5  # Balanced for XXLarge
        
        return new_params
    
    def _get_combination_key(self, params):
        """Generate a unique key for parameter combination to detect duplicates"""
        
        # Create a key from the most important parameters
        key_parts = [
            str(params.get('batch_size', '')),
            str(params.get('lr', '')),
            str(params.get('hidden_dim', '')),
            str(params.get('gcn_dim', '')),
            str(params.get('gcn_type', '')),
            str(params.get('attention_heads', '')),
            str(params.get('gcn_layers', '')),
            str(params.get('max_span_size', '')),
            str(params.get('max_pairs', '')),
            str(params.get('epochs', ''))
        ]
        
        return '|'.join(key_parts)
    
    def preview_combinations(self):
        """Preview all parameter combinations that will be tested"""
        
        print(f"🔍 Preview of {len(self.parameter_combinations)} parameter combinations:")
        print("="*80)
        
        for i, combo in enumerate(self.parameter_combinations):
            print(f"\n📋 Trial {i+1}: {combo['description']}")
            print("-" * 50)
            
            # Group parameters by category
            training_params = {k: v for k, v in combo.items() if k in ['batch_size', 'lr', 'lr_warmup', 'weight_decay', 'max_grad_norm', 'epochs']}
            model_params = {k: v for k, v in combo.items() if k in ['hidden_dim', 'gcn_dim', 'gcn_layers', 'attention_heads', 'lstm_layers', 'lstm_dim']}
            gcn_params = {k: v for k, v in combo.items() if k in ['gcn_type', 'use_residual', 'use_layer_norm', 'use_multi_scale', 'use_graph_attention']}
            memory_params = {k: v for k, v in combo.items() if k in ['max_span_size', 'max_pairs', 'neg_entity_count', 'neg_triple_count']}
            reg_params = {k: v for k, v in combo.items() if k in ['drop_out_rate', 'gcn_dropout', 'prop_drop', 'sen_filter_threshold']}
            
            if training_params:
                print("  🚀 Training:", " | ".join([f"{k}={v}" for k, v in training_params.items()]))
            if model_params:
                print("  🏗️  Model:", " | ".join([f"{k}={v}" for k, v in model_params.items()]))
            if gcn_params:
                print("  🌐 GCN:", " | ".join([f"{k}={v}" for k, v in gcn_params.items()]))
            if memory_params:
                print("  💾 Memory:", " | ".join([f"{k}={v}" for k, v in memory_params.items()]))
            if reg_params:
                print("  🛡️  Regularization:", " | ".join([f"{k}={v}" for k, v in reg_params.items()]))
        
        print(f"\n✅ Total combinations: {len(self.parameter_combinations)}")
        print(f"🎯 Trials to run: {self.n_trials}")
        print("="*80)
    
    def run_optimization(self):
        """Run the hyperparameter optimization"""
        
        print(f"Starting hyperparameter optimization for DeBERTa-v2-XXLarge")
        print(f"Dataset: {self.dataset}")
        print(f"Number of trials: {self.n_trials}")
        print(f"Results directory: {self.results_dir}")
        print("="*60)
        
        results = []
        best_score = -1.0
        best_params = None
        
        for trial_num, params in enumerate(self.parameter_combinations[:self.n_trials]):
            print(f"\nTRIAL {trial_num + 1}/{self.n_trials}")
            print(f"Description: {params['description']}")
            print(f"Parameters: {json.dumps({k: v for k, v in params.items() if k != 'description'}, indent=2)}")
            
            try:
                # Run training trial
                score = self._run_training_trial(params, trial_num + 1)
                
                # Store results
                trial_result = {
                    'trial_number': trial_num + 1,
                    'params': params,
                    'score': score,
                    'timestamp': time.time()
                }
                results.append(trial_result)
                
                # Update best score
                if score > best_score:
                    best_score = score
                    best_params = params
                
                print(f"Trial {trial_num + 1} completed with score: {score:.4f}")
                print(f"Best score so far: {best_score:.4f}")
                
            except Exception as e:
                print(f"Trial {trial_num + 1} failed: {str(e)}")
                results.append({
                    'trial_number': trial_num + 1,
                    'params': params,
                    'score': -1.0,
                    'error': str(e),
                    'timestamp': time.time()
                })
            
            # Save intermediate results
            self._save_results(results, best_score, best_params)
            
            print("-" * 40)
        
        # Final summary
        self._print_final_summary(results, best_score, best_params)
    
    def _run_training_trial(self, params, trial_num):
        """Run a single training trial"""
        
        # Create parameter file
        param_file = self.results_dir / f"trial_{trial_num}_params.py"
        self._create_parameter_file(param_file, params)
        
        # Debug: Check if parameter file exists
        if not param_file.exists():
            print(f"ERROR: Parameter file {param_file} was not created!")
            return -1.0
        
        try:
            # Create trial log directory
            trial_log_dir = self.results_dir / f"trial_{trial_num}"
            trial_log_dir.mkdir(exist_ok=True)
            
            # Run training
            cmd = [
                sys.executable, "train_improved.py",
                "--param_file", str(param_file),
                "--trial_log_dir", str(trial_log_dir)
            ]
            
            # Debug: Print the command being executed
            print(f"Executing command: {' '.join(cmd)}")
            print(f"Parameter file: {param_file}")
            print(f"Parameter file exists: {param_file.exists()}")
            
            # Run with real-time output
            print(f"Starting training for trial {trial_num}...")
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=1800,  # 30 minutes timeout
                cwd=os.getcwd(),
                bufsize=1,
                universal_newlines=True
            )
            
            # Print real-time output during training if verbose mode
            if self.verbose:
                print("Training Output:")
                print(result.stdout)
                if result.stderr:
                    print("Training Errors:")
                    print(result.stderr)
            
            if result.returncode != 0:
                print(f"Training failed with return code: {result.returncode}")
                print("Full training output:")
                print(result.stdout)
                return -1.0
            
            # Extract score
            score = self._extract_score(result.stdout)
            
            return score
            
        finally:
            # Clean up
            if param_file.exists():
                param_file.unlink()
    
    def _create_parameter_file(self, param_file, params):
        """Create a parameter file for the trial"""
        
        param_content = f'''# Auto-generated parameters for trial
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
    parser.add_argument("--dataset", default="{self.dataset}", type=str, help="Dataset to use")
    
    # Enhanced GCN parameters
    parser.add_argument("--use_improved_gcn", default=True, help="Use improved GCN modules")
    parser.add_argument("--gcn_type", default="{params['gcn_type']}", help="Type of GCN to use")
    parser.add_argument("--gcn_layers", type=int, default={params['gcn_layers']}, help="Number of GCN layers")
    parser.add_argument("--attention_heads", default={params['attention_heads']}, type=int, help="Number of attention heads")
    parser.add_argument("--use_residual", default={params['use_residual']}, help="Use residual connections")
    parser.add_argument("--use_layer_norm", default={params['use_layer_norm']}, help="Use layer normalization")
    parser.add_argument("--use_multi_scale", default={params.get('use_multi_scale', True)}, help="Use multi-scale features")
    parser.add_argument("--use_graph_attention", default={params.get('use_graph_attention', True)}, help="Use graph attention")
    parser.add_argument("--use_adaptive_edges", default={params.get('use_adaptive_edges', False)}, help="Use adaptive edges")
    parser.add_argument("--use_relative_position", default=True, help="Use relative position encoding")
    parser.add_argument("--use_global_context", default=True, help="Use global context modeling")
    parser.add_argument("--gcn_heads", default=8, type=int, help="Number of attention heads for GCN")
    parser.add_argument("--gcn_aggr", default="mean", help="Aggregation method for GraphSAGE")
    parser.add_argument("--gcn_eps", default=0.0, type=float, help="Epsilon for GIN convolution")
    parser.add_argument("--gcn_k", default=3, type=int, help="K parameter for Chebyshev GCN")
    
    # Model parameters
    parser.add_argument("--drop_out_rate", type=float, default={params['drop_out_rate']}, help="Dropout rate")
    parser.add_argument("--is_bidirect", default=True, help="Do use bi-RNN layer")
    parser.add_argument("--use_gated", default=False, help="Do use gcnconv and gatedgraphconv")
    parser.add_argument("--hidden_dim", type=int, default={params['hidden_dim']}, help="Hidden dimension")
    parser.add_argument("--emb_dim", type=int, default=1536, help="Embedding dimension")
    parser.add_argument("--lstm_layers", type=int, default=2, help="LSTM layers")
    parser.add_argument("--lstm_dim", type=int, default=384, help="LSTM dimension")
    parser.add_argument("--prefix", type=str, default="data/", help="Dataset and embedding path prefix")
    parser.add_argument("--span_generator", type=str, default="Max", help="Span generator option")
    parser.add_argument("--num_layers", type=int, default=2, help="Num of GCN layers")
    parser.add_argument("--mem_dim", type=int, default=768, help="Mutual biaffine mem dim")
    parser.add_argument("--gcn_dropout", type=float, default={params['gcn_dropout']}, help="GCN dropout")
    parser.add_argument("--pooling", default="avg", type=str, help="Pooling method")
    parser.add_argument("--gcn_dim", type=int, default={params['gcn_dim']}, help="GCN dimension")
    parser.add_argument("--deberta_feature_dim", type=int, default=1536, help="DeBERTa feature dimension")
    
    # Training parameters
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--max_span_size", type=int, default={params['max_span_size']}, help="Max span size")
    parser.add_argument("--lowercase", action="store_true", default=True, help="Case sensitive training")
    parser.add_argument("--max_pairs", type=int, default={params['max_pairs']}, help="Max entity pairs")
    parser.add_argument("--sen_filter_threshold", type=float, default={params['sen_filter_threshold']}, help="Sentiment filter threshold")
    parser.add_argument("--sampling_limit", type=int, default=100, help="Max samples in queue")
    parser.add_argument("--neg_entity_count", type=int, default={params['neg_entity_count']}, help="Negative entity count")
    parser.add_argument("--neg_triple_count", type=int, default={params['neg_triple_count']}, help="Negative triple count")
    parser.add_argument("--tokenizer_path", default="bert-base-uncased", type=str, help="Tokenizer load path")
    parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU even if CUDA available")
    parser.add_argument("--size_embedding", type=int, default=25, help="Dimensionality of size embedding")
    parser.add_argument("--sampling_processes", type=int, default=4, help="Number of sampling processes")
    parser.add_argument("--prop_drop", type=float, default={params['prop_drop']}, help="D2E2S dropout")
    parser.add_argument("--freeze_transformer", action="store_true", default=False, help="Freeze transformer parameters")
    
    # Optimization parameters
    parser.add_argument("--batch_size", type=int, default={params['batch_size']}, help="Batch size")
    parser.add_argument("--epochs", type=int, default={params['epochs']}, help="Number of epochs")
    parser.add_argument("--lr", type=float, default={params['lr']}, help="Learning rate")
    parser.add_argument("--lr_warmup", type=float, default={params['lr_warmup']}, help="LR warmup")
    parser.add_argument("--weight_decay", type=float, default={params['weight_decay']}, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default={params['max_grad_norm']}, help="Max gradient norm")
    
    # Other parameters
    parser.add_argument("--log_path", type=str, default="log/", help="Log directory path")
    parser.add_argument("--train_log_iter", type=int, default=1, help="Log training every x iterations")
    parser.add_argument("--pretrained_deberta_name", default="microsoft/deberta-v2-xxlarge", type=str)
    parser.add_argument("--init_eval", action="store_true", default=False, help="Evaluate before training")
    parser.add_argument("--final_eval", action="store_true", default=False, help="Evaluate only after training")
    parser.add_argument("--store_predictions", action="store_true", default=True, help="Store predictions")
    parser.add_argument("--store_examples", action="store_true", default=True, help="Store examples")
    parser.add_argument("--example_count", type=int, default=None, help="Count of evaluation examples")
    parser.add_argument("--save_path", type=str, default="data/save/", help="Model checkpoint path")
    parser.add_argument("--save_optimizer", action="store_true", default=False, help="Save optimizer")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--deterministic", action="store_true", default=False, help="Deterministic mode")
    
    opt = parser.parse_args([])
    opt.label = opt.dataset
    opt.dataset_file = dataset_files[opt.dataset]
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return opt
'''
        
        with open(param_file, 'w') as f:
            f.write(param_content)
    
    def _extract_score(self, output):
        """Extract F1 score from training output"""
        
        try:
            lines = output.split('\n')
            for line in lines:
                if "Best F1 score:" in line:
                    score_str = line.split("Best F1 score:")[1].split("at")[0].strip()
                    return float(score_str)
            
            # Try to extract from log files
            return self._extract_score_from_logs()
            
        except Exception as e:
            print(f"Error extracting score: {str(e)}")
            return -1.0
    
    def _extract_score_from_logs(self):
        """Extract score from log files"""
        
        try:
            log_dirs = list(Path("log").glob("*/result*.txt"))
            if not log_dirs:
                return -1.0
            
            latest_result = max(log_dirs, key=lambda x: x.stat().st_mtime)
            
            with open(latest_result, 'r') as f:
                content = f.read()
                if "mic_f1_score" in content:
                    lines = content.split('\n')
                    for line in lines:
                        if "mic_f1_score" in line:
                            try:
                                score_str = line.split(":")[1].strip().strip("'")
                                return float(score_str)
                            except:
                                continue
            
            return -1.0
            
        except Exception as e:
            print(f"Error extracting score from logs: {str(e)}")
            return -1.0
    
    def _save_results(self, results, best_score, best_params):
        """Save optimization results"""
        
        summary = {
            'dataset': self.dataset,
            'total_trials': len(results),
            'best_score': best_score,
            'best_params': best_params,
            'trials': results
        }
        
        # Save JSON
        summary_file = self.results_dir / "optimization_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save readable text
        summary_txt = self.results_dir / "optimization_summary.txt"
        with open(summary_txt, 'w') as f:
            f.write(f"Hyperparameter Optimization Summary\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Dataset: {self.dataset}\n")
            f.write(f"Total Trials: {len(results)}\n")
            f.write(f"Best Score: {best_score:.4f}\n\n")
            
            if best_params:
                f.write(f"Best Parameters:\n")
                for key, value in best_params.items():
                    if key != 'description':
                        f.write(f"  {key}: {value}\n")
            
            f.write(f"\nTrial Results:\n")
            for trial in results:
                f.write(f"  Trial {trial['trial_number']}: {trial['score']:.4f}")
                if 'error' in trial:
                    f.write(f" (Error: {trial['error']})")
                f.write(f"\n")
    
    def _print_final_summary(self, results, best_score, best_params):
        """Print final optimization summary"""
        
        print(f"\n{'='*60}")
        print(f"OPTIMIZATION COMPLETED")
        print(f"{'='*60}")
        print(f"Dataset: {self.dataset}")
        print(f"Total Trials: {len(results)}")
        print(f"Best F1 Score: {best_score:.4f}")
        
        if best_params:
            print(f"\nBest Parameters:")
            for key, value in best_params.items():
                if key != 'description':
                    print(f"  {key}: {value}")
        
        print(f"\nResults saved to: {self.results_dir}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Simple hyperparameter tuning for DeBERTa-v2-XXLarge")
    parser.add_argument("--dataset", default="14res", choices=["14res", "15res", "16res", "14lap"], 
                       help="Dataset to use for optimization")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of trials to run")
    parser.add_argument("--config", default="hyperparameter_config.py", 
                       help="Path to Python configuration file")
    parser.add_argument("--preview", action="store_true", 
                       help="Preview parameter combinations without running optimization")
    parser.add_argument("--verbose", action="store_true", 
                       help="Show detailed epoch-by-epoch training progress")
    
    args = parser.parse_args()
    
    # Create tuner
    tuner = SimpleHyperparameterTuner(
        dataset=args.dataset, 
        n_trials=args.n_trials,
        config_file=args.config,
        verbose=args.verbose
    )
    
    if args.preview:
        # Only preview combinations
        tuner.preview_combinations()
    else:
        # Preview and then run optimization
        tuner.preview_combinations()
        tuner.run_optimization()


if __name__ == "__main__":
    main()
