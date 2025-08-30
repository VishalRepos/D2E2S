# Hyperparameter Tuner with Real-time Output Display

This document explains how to use the improved hyperparameter tuner that now displays real-time training progress, including epoch-by-epoch F1 scores and charts.

## 🚀 Features

- **Real-time Output**: See training progress as it happens, including:
  - Epoch-by-epoch F1 scores
  - Progress bars
  - Training loss
  - Evaluation results
- **Visual Progress Indicators**: Emoji-based status indicators for better readability
- **Automatic Score Extraction**: Automatically extracts the best F1 score from training output
- **Comprehensive Logging**: Saves all results to JSON and text files
- **Timeout Protection**: Prevents trials from running indefinitely
- **Error Handling**: Graceful handling of failed trials

## 📋 Usage

### Basic Usage

```bash
# Run with default settings (20 trials on 14res dataset)
python simple_hyperparameter_tuner.py

# Run with custom settings
python simple_hyperparameter_tuner.py --dataset 15res --n_trials 10 --verbose

# Preview parameter combinations without running
python simple_hyperparameter_tuner.py --preview
```

### Command Line Options

- `--dataset`: Dataset to use (14res, 15res, 16res, 14lap) [default: 14res]
- `--n_trials`: Number of trials to run [default: 20]
- `--config`: Path to configuration file [default: hyperparameter_config.py]
- `--preview`: Preview parameter combinations without running optimization
- `--verbose`: Show detailed training output

### Test Script

For testing purposes, you can use the test script:

```bash
python test_hyperparameter_tuner.py
```

This will run just 2 trials to verify everything works correctly.

## 📊 What You'll See

### During Training

```
🎯 TRIAL 1/20
📝 Description: Random trial 1
🏆 Current Best Score: 0.0000
============================================================
🔧 Key Parameters: {'batch_size': 8, 'lr': 0.0001, 'epochs': 10, 'gcn_type': 'improved', 'gcn_layers': 3, 'attention_heads': 8}
----------------------------------------
🚀 Starting training for trial 1...
============================================================
📊 Training Progress (Real-time):
----------------------------------------
   Using improved GCN modules with configuration:
   - GCN Type: improved
   - GCN Layers: 3
   - Attention Heads: 8
   - Use Residual: True
   - Use Layer Norm: True
   - Use Multi-scale: True
   - Use Graph Attention: True
   
   Train epoch 0: 100%|██████████| 50/50 [02:30<00:00,  3.33it/s]
   🔍 EVALUATION: Evaluate epoch 1: 100%|██████████| 12/12 [00:15<00:00,  1.25it/s]
   🔍 EVALUATION: No. 1 ：....
   🔍 EVALUATION: ner_entity: 
   🔍 EVALUATION: {'mic_precision': 0.0, 'mic_recall': 0.0, 'mic_f1_score': 0.0, ...}
   🔍 EVALUATION: rec: 
   🔍 EVALUATION: {'mic_precision': 0.8234, 'mic_recall': 0.7891, 'mic_f1_score': 0.8059, ...}
   
   Train epoch 1: 100%|██████████| 50/50 [02:28<00:00,  3.37it/s]
   ...
   
   Best F1 score: 0.8234 at epoch 5

----------------------------------------
✅ Training completed for trial 1

📊 EVALUATION RESULTS for Trial 1:
==================================================
No. 1 ：....
ner_entity: 
{'mic_precision': 0.0, 'mic_recall': 0.0, 'mic_f1_score': 0.0, ...}
rec: 
{'mic_precision': 0.8234, 'mic_recall': 0.7891, 'mic_f1_score': 0.8059, ...}

🏁 FINAL TRAINING SUMMARY:
==================================================
Best F1 score: 0.8234 at epoch 5

🏆 Final Best F1 Score: 0.8234
🎉 NEW BEST SCORE! Trial 1: 0.8234
🏆 Best score so far: 0.8234
⏱️  Trial 1 completed in 125.3 seconds
📈 Progress: 1/20 trials completed
```

### Final Summary

```
============================================================
🎉 OPTIMIZATION COMPLETED! 🎉
============================================================
📊 Dataset: 14res
🎯 Total Trials: 20
🏆 Best F1 Score: 0.8567
✅ Success Rate: 95.0% (19/20 trials)

🏆 Best Parameters:
  🚀 Training: {'batch_size': 16, 'lr': 0.0002, 'lr_warmup': 0.15, 'weight_decay': 0.01, 'epochs': 15}
  🏗️  Model: {'hidden_dim': 768, 'gcn_dim': 768, 'gcn_layers': 4, 'attention_heads': 12}
  🌐 GCN: {'gcn_type': 'improved', 'use_residual': True, 'use_layer_norm': True, 'use_multi_scale': True}

📁 Results saved to: hyperparameter_results
📄 Summary files: optimization_summary.json, optimization_summary.txt
============================================================
```

## 📁 Output Files

The tuner creates several output files in the `hyperparameter_results` directory:

- `optimization_summary.json`: Complete results in JSON format
- `optimization_summary.txt`: Human-readable summary
- `trial_X_params.py`: Parameter files for each trial
- `trial_X/`: Log directories for each trial

## 🔧 Configuration

You can customize the hyperparameter search space by modifying `hyperparameter_config.py`:

```python
def get_config():
    return {
        'search_spaces': {
            'batch_size': {
                'type': 'categorical',
                'values': [4, 8, 16, 32]
            },
            'lr': {
                'type': 'float',
                'low': 1e-5,
                'high': 1e-3,
                'log': True
            },
            # ... more parameters
        },
        'predefined_combinations': [
            # Add specific combinations to test
        ]
    }
```

## ⚠️ Troubleshooting

### Common Issues

1. **No real-time output**: Make sure you're not using `--preview` mode
2. **Training fails**: Check GPU memory and reduce batch size
3. **Score extraction fails**: Check that training completed successfully
4. **Timeout errors**: Increase timeout in the code or reduce epochs

### Memory Issues

If you encounter GPU memory issues:

1. Reduce `batch_size` in the search space
2. Reduce `max_span_size` and `max_pairs`
3. Use smaller model dimensions
4. Enable gradient checkpointing

## 🎯 Tips for Best Results

1. **Start with fewer trials** (5-10) to test your setup
2. **Use the preview mode** to check parameter combinations
3. **Monitor GPU memory** during training
4. **Check the logs** in `hyperparameter_results/` for detailed information
5. **Use the test script** first to verify everything works

## 📈 Performance

- Each trial typically takes 10-30 minutes depending on:
  - Number of epochs
  - Batch size
  - Dataset size
  - GPU performance
- Total optimization time: 2-10 hours for 20 trials
- Memory usage: 8-16GB GPU memory recommended
