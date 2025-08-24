#📋 New Preview Feature
```
# See what combinations will be tested (without running)
python simple_hyperparameter_tuner.py --dataset 14res --n_trials 5 --preview
```

## This will show you:
- Trial 1: Conservative settings (batch_size=4, lr=1e-5, gcn_type='improved')
- Trial 2: Balanced settings (batch_size=6, lr=2e-5, gcn_type='gatv2')
- Trial 3: Performance settings (batch_size=8, lr=3e-5, gcn_type='hybrid')
- Trial 4: Random variation 1 (unique combination)
- Trial 5: Random variation 2 (unique combination)

# 🚀 Usage Examples
```
# Test 3 unique combinations
python simple_hyperparameter_tuner.py --dataset 14res --n_trials 3

# Test 20 unique combinations  
python simple_hyperparameter_tuner.py --dataset 14res --n_trials 20

# Test 50 unique combinations
python simple_hyperparameter_tuner.py --dataset 14res --n_trials 50
```
