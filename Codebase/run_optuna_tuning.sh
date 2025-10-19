#!/bin/bash

# Quick Start Script for Optuna Hyperparameter Tuning
# Usage: ./run_optuna_tuning.sh [dataset] [strategy] [trials]

set -e

# Default parameters
DATASET=${1:-"14res"}
STRATEGY=${2:-"balanced"}
TRIALS=${3:-50}

echo "🚀 Starting Optuna Hyperparameter Tuning"
echo "📊 Dataset: $DATASET"
echo "🎯 Strategy: $STRATEGY"
echo "🎲 Trials: $TRIALS"
echo "=================================="

# Install requirements if needed
echo "📦 Installing requirements..."
pip install -r requirements.txt

# Create database storage
STORAGE="sqlite:///optuna_results/optuna_study_${DATASET}_${STRATEGY}.db"
mkdir -p optuna_results

echo "🔬 Running optimization..."
python advanced_optuna_tuner.py \
    --dataset $DATASET \
    --strategy $STRATEGY \
    --n_trials $TRIALS \
    --storage $STORAGE \
    --sampler tpe

echo "✅ Optimization completed!"
echo "📊 To view results in dashboard:"
echo "   optuna-dashboard $STORAGE"
echo "   Then open: http://localhost:8080"