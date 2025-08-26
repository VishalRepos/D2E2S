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

# Search Spaces for DeBERTa-v2-XXLarge - Most Relevant Parameters Only
SEARCH_SPACES = {
    # Training Parameters (Critical for XXLarge)
    'batch_size': {
        'type': 'categorical',
        'values': [4, 6, 8],  # Reduced for XXLarge memory constraints
        'description': 'Training batch size for DeBERTa-v2-XXLarge memory optimization'
    },
    
    'lr': {
        'type': 'float',
        'low': 1e-5,
        'high': 3e-5,  # Reduced range for XXLarge stability
        'log': True,
        'description': 'Learning rate for DeBERTa-v2-XXLarge (conservative range)'
    },
    
    # GCN Architecture (Most Impactful)
    'gcn_type': {
        'type': 'categorical',
        # 'values': ['improved', 'gatv2', 'hybrid'],  # Most proven for XXLarge
        'values': ['hybrid'],  # Most proven for XXLarge
        'description': 'GCN type optimized for DeBERTa-v2-XXLarge'
    },
    
    'gcn_layers': {
        'type': 'int',
        'low': 2,
        'high': 4,  # Reduced for XXLarge efficiency
        'description': 'Number of GCN layers (optimized for XXLarge)'
    },
    
    'attention_heads': {
        'type': 'categorical',
        'values': [8, 12],  # Reduced for XXLarge memory
        'description': 'Attention heads (memory-optimized for XXLarge)'
    },
    
    # Memory Management (Critical for XXLarge)
    'max_pairs': {
        'type': 'categorical',
        'values': [400, 600, 800],  # Reduced for XXLarge memory
        'description': 'Max entity pairs (XXLarge memory optimized)'
    },
    
    # Fixed Training Strategy
    'epochs': {
        'type': 'int',
        'low': 40,
        'high': 40,
        'description': 'Fixed epochs for consistent XXLarge training'
    }
}

# Additional fixed parameters for DeBERTa-v2-XXLarge
FIXED_PARAMS = {
    'emb_dim': 1536,  # DeBERTa-v2-XXLarge feature dimension
    'hidden_dim': 768,  # Half of emb_dim for bidirectional LSTM
    'deberta_feature_dim': 1536,  # DeBERTa-v2-XXLarge output dimension
    'gcn_dim': 768,  # Optimized for XXLarge
    'max_span_size': 6,  # Memory efficient for XXLarge
    'neg_entity_count': 50,  # Reduced for XXLarge memory
    'neg_triple_count': 50,  # Reduced for XXLarge memory
    'use_residual': True,  # Proven for XXLarge
    'use_layer_norm': True,  # Essential for XXLarge
    'use_multi_scale': True,  # Proven for XXLarge
    'use_graph_attention': True,  # Proven for XXLarge
    'drop_out_rate': 0.3,  # Conservative for XXLarge
    'gcn_dropout': 0.1,  # Conservative for XXLarge
    'prop_drop': 0.05,  # Conservative for XXLarge
    'sen_filter_threshold': 0.5,  # Balanced for XXLarge
    'lr_warmup': 0.15,  # Conservative for XXLarge
    'weight_decay': 0.01,  # Standard regularization
    'max_grad_norm': 1.0,  # Stable for XXLarge
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

# Random Parameter Generation (No predefined combinations)
PREDEFINED_COMBINATIONS = []

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
