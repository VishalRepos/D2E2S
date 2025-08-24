# Hyperparameter Tuning Configuration for DeBERTa-v2-XXLarge
# This file defines the search spaces and optimization settings as Python variables

# Study Configuration
STUDY_NAME = "deberta_xxlarge_optimization"
STUDY_DIRECTION = "maximize"  # Maximize F1 score
STUDY_SAMPLER = "tpe"  # Tree-structured Parzen Estimator
STUDY_PRUNER = "median"  # Median pruner for early stopping

# Optimization Settings
N_TRIALS = 50
TIMEOUT = 7200  # 2 hours in seconds
N_STARTUP_TRIALS = 5  # Random trials before TPE
N_WARMUP_STEPS = 10  # Warmup steps for pruner

# Search Spaces for Critical Parameters
SEARCH_SPACES = {
    # Training Parameters (Most Important)
    'batch_size': {
        'type': 'categorical',
        'values': [4, 6, 8, 10, 12],
        'description': 'Training batch size for memory optimization'
    },
    
    'lr': {
        'type': 'float',
        'low': 1e-5,
        'high': 5e-5,
        'log': True,
        'description': 'Learning rate for DeBERTa-v2-XXLarge'
    },
    
    'lr_warmup': {
        'type': 'float',
        'low': 0.1,
        'high': 0.3,
        'description': 'Learning rate warmup proportion'
    },
    
    'weight_decay': {
        'type': 'float',
        'low': 0.001,
        'high': 0.05,
        'log': True,
        'description': 'Weight decay for regularization'
    },
    
    'max_grad_norm': {
        'type': 'float',
        'low': 0.5,
        'high': 2.0,
        'description': 'Maximum gradient norm for clipping'
    },
    
    # Model Architecture
    'hidden_dim': {
        'type': 'categorical',
        'values': [768, 1024, 1536],
        'description': 'Hidden layer dimension'
    },
    
    'gcn_dim': {
        'type': 'categorical',
        'values': [256, 384, 512, 768],
        'description': 'GCN layer dimension'
    },
    
    'gcn_layers': {
        'type': 'int',
        'low': 3,
        'high': 6,
        'description': 'Number of GCN layers'
    },
    
    'attention_heads': {
        'type': 'categorical',
        'values': [8, 12, 16],
        'description': 'Number of attention heads'
    },
    
    'lstm_layers': {
        'type': 'int',
        'low': 1,
        'high': 3,
        'description': 'Number of LSTM layers'
    },
    
    'lstm_dim': {
        'type': 'categorical',
        'values': [256, 384, 512],
        'description': 'LSTM cell dimension'
    },
    
    # Memory Optimization
    'max_span_size': {
        'type': 'int',
        'low': 5,
        'high': 8,
        'description': 'Maximum span size for memory efficiency'
    },
    
    'max_pairs': {
        'type': 'categorical',
        'values': [400, 600, 800, 1000],
        'description': 'Maximum entity pairs to process'
    },
    
    'neg_entity_count': {
        'type': 'categorical',
        'values': [50, 75, 100],
        'description': 'Negative entity samples count'
    },
    
    'neg_triple_count': {
        'type': 'categorical',
        'values': [50, 75, 100],
        'description': 'Negative triple samples count'
    },
    
    # GCN Architecture
    'gcn_type': {
        'type': 'categorical',
        'values': ['improved', 'gatv2', 'hybrid', 'dynamic'],
        'description': 'Type of GCN to use'
    },
    
    'use_residual': {
        'type': 'categorical',
        'values': [True, False],
        'description': 'Use residual connections'
    },
    
    'use_layer_norm': {
        'type': 'categorical',
        'values': [True, False],
        'description': 'Use layer normalization'
    },
    
    'use_multi_scale': {
        'type': 'categorical',
        'values': [True, False],
        'description': 'Use multi-scale feature aggregation'
    },
    
    'use_graph_attention': {
        'type': 'categorical',
        'values': [True, False],
        'description': 'Use graph attention mechanism'
    },
    
    # Regularization
    'drop_out_rate': {
        'type': 'float',
        'low': 0.3,
        'high': 0.7,
        'description': 'Main dropout rate'
    },
    
    'gcn_dropout': {
        'type': 'float',
        'low': 0.1,
        'high': 0.4,
        'description': 'GCN-specific dropout'
    },
    
    'prop_drop': {
        'type': 'float',
        'low': 0.05,
        'high': 0.2,
        'description': 'D2E2S dropout probability'
    },
    
    # Training Strategy
    'epochs': {
        'type': 'int',
        'low': 80,
        'high': 120,
        'description': 'Number of training epochs'
    },
    
    'sen_filter_threshold': {
        'type': 'float',
        'low': 0.3,
        'high': 0.6,
        'description': 'Sentiment filter threshold'
    }
}

# Dataset Configuration
DATASETS = ["14res", "15res", "16res", "14lap"]

# Resource Constraints
RESOURCE_CONSTRAINTS = {
    'gpu_memory': "24GB",  # Minimum GPU memory required
    'max_trial_time': 1800,  # 30 minutes per trial
    'max_concurrent_trials': 1  # Sequential trials for memory management
}

# Early Stopping
EARLY_STOPPING = {
    'enabled': True,
    'patience': 10,  # Stop if no improvement for 10 trials
    'min_trials': 15  # Minimum trials before early stopping
}

# Result Storage
RESULT_STORAGE = {
    'save_models': False,  # Don't save all models to save space
    'save_logs': True,  # Save training logs
    'save_predictions': False,  # Don't save all predictions
    'save_examples': False  # Don't save all examples
}

# Visualization
VISUALIZATION = {
    'plot_optimization_history': True,
    'plot_param_importances': True,
    'plot_parallel_coordinate': True,
    'plot_contour': True
}

# Predefined Parameter Combinations for Quick Start
PREDEFINED_COMBINATIONS = [
    # Conservative settings (memory-efficient)
    {
        'batch_size': 4,
        'lr': 1e-5,
        'lr_warmup': 0.15,
        'weight_decay': 0.01,
        'max_grad_norm': 1.0,
        'hidden_dim': 768,
        'gcn_dim': 384,
        'gcn_layers': 3,
        'attention_heads': 8,
        'max_span_size': 6,
        'max_pairs': 600,
        'neg_entity_count': 75,
        'neg_triple_count': 75,
        'gcn_type': 'improved',
        'use_residual': True,
        'use_layer_norm': True,
        'drop_out_rate': 0.5,
        'gcn_dropout': 0.2,
        'epochs': 100,
        'description': 'Conservative memory-efficient settings'
    },
    
    # Balanced settings
    {
        'batch_size': 6,
        'lr': 2e-5,
        'lr_warmup': 0.2,
        'weight_decay': 0.01,
        'max_grad_norm': 1.0,
        'hidden_dim': 1024,
        'gcn_dim': 512,
        'gcn_layers': 4,
        'attention_heads': 12,
        'max_span_size': 7,
        'max_pairs': 800,
        'neg_entity_count': 75,
        'neg_triple_count': 75,
        'gcn_type': 'gatv2',
        'use_residual': True,
        'use_layer_norm': True,
        'drop_out_rate': 0.4,
        'gcn_dropout': 0.15,
        'epochs': 100,
        'description': 'Balanced performance settings'
    },
    
    # Performance-focused settings
    {
        'batch_size': 8,
        'lr': 3e-5,
        'lr_warmup': 0.25,
        'weight_decay': 0.005,
        'max_grad_norm': 0.8,
        'hidden_dim': 1536,
        'gcn_dim': 768,
        'gcn_layers': 5,
        'attention_heads': 16,
        'max_span_size': 8,
        'max_pairs': 1000,
        'neg_entity_count': 100,
        'neg_triple_count': 100,
        'gcn_type': 'hybrid',
        'use_residual': True,
        'use_layer_norm': True,
        'drop_out_rate': 0.3,
        'gcn_dropout': 0.1,
        'epochs': 120,
        'description': 'Performance-focused settings'
    },
    
    # GATv2 focused
    {
        'batch_size': 6,
        'lr': 2e-5,
        'lr_warmup': 0.2,
        'weight_decay': 0.01,
        'max_grad_norm': 1.0,
        'hidden_dim': 1024,
        'gcn_dim': 512,
        'gcn_layers': 4,
        'attention_heads': 12,
        'max_span_size': 7,
        'max_pairs': 800,
        'neg_entity_count': 75,
        'neg_triple_count': 75,
        'gcn_type': 'gatv2',
        'use_residual': True,
        'use_layer_norm': True,
        'use_multi_scale': True,
        'use_graph_attention': True,
        'drop_out_rate': 0.4,
        'gcn_dropout': 0.15,
        'epochs': 100,
        'description': 'GATv2 with multi-scale attention'
    },
    
    # Dynamic GCN settings
    {
        'batch_size': 6,
        'lr': 2e-5,
        'lr_warmup': 0.2,
        'weight_decay': 0.01,
        'max_grad_norm': 1.0,
        'hidden_dim': 1024,
        'gcn_dim': 512,
        'gcn_layers': 4,
        'attention_heads': 12,
        'max_span_size': 7,
        'max_pairs': 800,
        'neg_entity_count': 75,
        'neg_triple_count': 75,
        'gcn_type': 'dynamic',
        'use_residual': True,
        'use_layer_norm': True,
        'use_adaptive_edges': True,
        'drop_out_rate': 0.4,
        'gcn_dropout': 0.15,
        'epochs': 100,
        'description': 'Dynamic GCN with adaptive edges'
    }
]

# Function to get configuration
def get_config():
    """Return the complete configuration dictionary"""
    return {
        'study': {
            'name': STUDY_NAME,
            'direction': STUDY_DIRECTION,
            'sampler': STUDY_SAMPLER,
            'pruner': STUDY_PRUNER
        },
        'optimization': {
            'n_trials': N_TRIALS,
            'timeout': TIMEOUT,
            'n_startup_trials': N_STARTUP_TRIALS,
            'n_warmup_steps': N_WARMUP_STEPS
        },
        'search_spaces': SEARCH_SPACES,
        'datasets': DATASETS,
        'resources': RESOURCE_CONSTRAINTS,
        'early_stopping': EARLY_STOPPING,
        'results': RESULT_STORAGE,
        'visualization': VISUALIZATION,
        'predefined_combinations': PREDEFINED_COMBINATIONS
    }

# Function to get specific configuration section
def get_search_spaces():
    """Return the search spaces configuration"""
    return SEARCH_SPACES

def get_predefined_combinations():
    """Return the predefined parameter combinations"""
    return PREDEFINED_COMBINATIONS

def get_optimization_settings():
    """Return the optimization settings"""
    return {
        'n_trials': N_TRIALS,
        'timeout': TIMEOUT,
        'n_startup_trials': N_STARTUP_TRIALS,
        'n_warmup_steps': N_WARMUP_STEPS
    }

# Function to validate configuration
def validate_config():
    """Validate the configuration and return any issues"""
    issues = []
    
    # Check required fields
    required_fields = ['batch_size', 'lr', 'hidden_dim', 'gcn_dim']
    for field in required_fields:
        if field not in SEARCH_SPACES:
            issues.append(f"Missing required search space: {field}")
    
    # Check value ranges
    if SEARCH_SPACES['lr']['low'] >= SEARCH_SPACES['lr']['high']:
        issues.append("Learning rate low value must be less than high value")
    
    if SEARCH_SPACES['batch_size']['values'] and min(SEARCH_SPACES['batch_size']['values']) <= 0:
        issues.append("Batch size values must be positive")
    
    return issues

# Function to print configuration summary
def print_config_summary():
    """Print a summary of the current configuration"""
    print(f"Hyperparameter Configuration Summary")
    print(f"{'='*50}")
    print(f"Study Name: {STUDY_NAME}")
    print(f"Direction: {STUDY_DIRECTION}")
    print(f"Number of Trials: {N_TRIALS}")
    print(f"Timeout: {TIMEOUT} seconds")
    print(f"Datasets: {', '.join(DATASETS)}")
    print(f"Predefined Combinations: {len(PREDEFINED_COMBINATIONS)}")
    print(f"Search Spaces: {len(SEARCH_SPACES)} parameters")
    
    # Print search space summary
    print(f"\nSearch Spaces:")
    for param, config in SEARCH_SPACES.items():
        if config['type'] == 'categorical':
            print(f"  {param}: {config['type']} with {len(config['values'])} values")
        elif config['type'] == 'float':
            print(f"  {param}: {config['type']} [{config['low']:.2e}, {config['high']:.2e}]")
        elif config['type'] == 'int':
            print(f"  {param}: {config['type']} [{config['low']}, {config['high']}]")

if __name__ == "__main__":
    # Print configuration summary when run directly
    print_config_summary()
    
    # Validate configuration
    issues = validate_config()
    if issues:
        print(f"\nConfiguration Issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print(f"\nConfiguration is valid!")
