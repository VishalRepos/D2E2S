#!/bin/bash
# Activation script for D2E2S virtual environment

echo "🚀 Activating D2E2S Virtual Environment..."
source d2e2s_env/bin/activate

echo "✅ Environment activated!"
echo "📦 Installed packages:"
echo "   - Optuna $(python -c 'import optuna; print(optuna.__version__)')"
echo "   - PyTorch $(python -c 'import torch; print(torch.__version__)')"
echo "   - Transformers $(python -c 'import transformers; print(transformers.__version__)')"

echo ""
echo "🎯 Ready for Optuna hyperparameter tuning!"
echo "💡 Usage examples:"
echo "   python advanced_optuna_tuner.py --dataset 14res --strategy quick --n_trials 30"
echo "   ./run_optuna_tuning.sh 14res balanced 50"
echo ""