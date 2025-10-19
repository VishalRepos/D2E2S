"""
Optuna Configuration for D2E2S Hyperparameter Optimization
Defines search spaces and optimization strategies
"""

def get_search_space_config():
    """Get search space configuration for different optimization strategies"""
    
    return {
        # Quick optimization (20-30 trials)
        'quick': {
            'batch_size': [8, 16],
            'lr': (1e-5, 1e-3, 'log'),
            'lr_warmup': (0.1, 0.2),
            'weight_decay': (0.005, 0.02),
            'gcn_type': ['improved', 'adaptive', 'hybrid'],
            'gcn_layers': (2, 3),
            'attention_heads': [8, 12],
            'epochs': (20, 40)
        },
        
        # Balanced optimization (50-100 trials)
        'balanced': {
            'batch_size': [4, 8, 16, 32],
            'lr': (1e-6, 1e-3, 'log'),
            'lr_warmup': (0.05, 0.3),
            'weight_decay': (1e-4, 1e-1, 'log'),
            'gcn_type': ['improved', 'adaptive', 'gatv2', 'hybrid'],
            'gcn_layers': (2, 4),
            'attention_heads': [4, 8, 12, 16],
            'hidden_dim': [384, 512, 768],
            'gcn_dim': [512, 768, 1024],
            'epochs': (20, 120)
        },
        
        # Comprehensive optimization (100+ trials)
        'comprehensive': {
            'batch_size': [4, 8, 16, 32],
            'lr': (1e-6, 1e-3, 'log'),
            'lr_warmup': (0.05, 0.3),
            'weight_decay': (1e-4, 1e-1, 'log'),
            'gcn_type': ['improved', 'adaptive', 'gatv2', 'gcn', 'sage', 'hybrid'],
            'gcn_layers': (2, 4),
            'attention_heads': [4, 8, 12, 16, 20],
            'hidden_dim': [256, 384, 512, 768, 1024],
            'gcn_dim': [300, 512, 768, 1024],
            'drop_out_rate': (0.1, 0.5),
            'gcn_dropout': (0.05, 0.3),
            'prop_drop': (0.01, 0.2),
            'max_span_size': (4, 10),
            'max_pairs': [500, 800, 1000, 1500],
            'use_residual': [True, False],
            'use_layer_norm': [True, False],
            'use_multi_scale': [True, False],
            'epochs': (20, 120)
        }
    }

def get_pruning_config():
    """Get pruning configuration for early stopping"""
    
    return {
        'median_pruner': {
            'n_startup_trials': 5,
            'n_warmup_steps': 10,
            'interval_steps': 5
        },
        'hyperband_pruner': {
            'min_resource': 10,
            'max_resource': 120,
            'reduction_factor': 3
        }
    }

def get_sampler_config():
    """Get sampler configuration for different optimization strategies"""
    
    return {
        'tpe': {
            'n_startup_trials': 10,
            'n_ei_candidates': 24
        },
        'cmaes': {
            'n_startup_trials': 10
        },
        'random': {}
    }