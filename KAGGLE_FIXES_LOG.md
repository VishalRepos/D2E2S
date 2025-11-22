# Kaggle Training Fixes - Change Log

## Fix 8: Default hidden_dim Parameter (2025-11-22 16:50)

### Issue Found
Debug output showed:
```
hidden_dim: 768  ← Using default, not command line parameter!
hidden_dim * 2 (bidirectional): 1536
```

The `--hidden_dim 384` parameter in the notebook was being **ignored**.

### Root Cause
Command line parameters weren't overriding the default value in `Parameter_Improved.py`.

### Solution
Changed the default value in `Parameter_Improved.py`:
```python
# Before
"--hidden_dim", type=int, default=768

# After  
"--hidden_dim", type=int, default=384
```

### Commit
`b481af6` - "Fix: Change default hidden_dim to 384 for deberta-v3-base"

### Expected Result
- `hidden_dim: 384`
- `hidden_dim * 2: 768` (matches GCN input)
- No more dimension mismatch errors

### Testing
- [ ] Restart Kaggle kernel
- [ ] Run All
- [ ] Verify debug shows `hidden_dim: 384`
- [ ] Verify training starts successfully

---

## Current State (2025-11-22 16:43)

### Issue
Training fails with matrix multiplication error:
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (280x1536 and 768x768)
```

### Root Cause
- LSTM is bidirectional and outputs `hidden_dim × 2`
- With `hidden_dim=768`: output = 1536 dimensions
- GCN expects 768 dimensions
- **Mismatch!**

### Current Configuration
- **Model**: `microsoft/deberta-v3-base` (768 dims)
- **Parameters Set**:
  - `--emb_dim 768`
  - `--hidden_dim 384` ← Should make LSTM output 768
  - `--deberta_feature_dim 768`
  - `--gcn_dim 768`
  - `--mem_dim 768`

### Problem
Despite setting `--hidden_dim 384`, LSTM still outputs 1536 dimensions.
This suggests the parameter isn't being used or there's a caching issue.

### Latest Changes
**Commit**: `622c703` - "Add comprehensive debugging for dimension tracking"

**Added Debugging**:
1. Notebook Step 2 now shows:
   - Git commit hash
   - Current directory
   - Parameter_Improved.py default values
   
2. train.py now prints:
   - All dimension parameters at startup
   - Model name
   - Calculated bidirectional output size

### Next Steps
1. Run notebook on Kaggle with debugging
2. Verify commit hash is `622c703`
3. Check if `hidden_dim` shows as 384 or 768
4. Based on output, determine if:
   - Parameter isn't being passed correctly
   - Default value is overriding it
   - There's a different issue

---

## Previous Fixes Applied

### Fix 1: Model Size (Memory Issue)
**Problem**: DeBERTa-v2-xxlarge (1.5B params) exceeded T4 GPU memory (15GB)
**Solution**: Changed to `microsoft/deberta-v3-base` (184M params)
**Commit**: `ab06ffe`

### Fix 2: Config Mismatch
**Problem**: Loading xxlarge config with v3-base weights
**Solution**: Use `AutoConfig.from_pretrained(args.pretrained_deberta_name)`
**Commit**: `5b802b3`

### Fix 3: Hardcoded Model in D2E2S_Model.py
**Problem**: Model file had hardcoded `deberta-v2-xxlarge`
**Solution**: Use `AutoModel.from_config(config)`
**Commit**: `07f6651`

### Fix 4: Dimension Parameters
**Problem**: All dimensions defaulted to 1536 (xxlarge size)
**Solution**: Added explicit parameters for 768 dimensions
**Commit**: `5cb88e8`

### Fix 5: Directory Navigation
**Problem**: Nested directory structure causing old code to run
**Solution**: Clean clone with `rm -rf` and proper navigation
**Commit**: `dabb910`

### Fix 6: Bidirectional LSTM Output
**Problem**: LSTM outputs `hidden_dim × 2` due to bidirectional
**Solution**: Set `hidden_dim=384` so output is 768
**Commit**: `fca1564`

### Fix 7: Memory Optimizations
- Gradient accumulation (4 steps)
- Reduced batch size (2 → 8)
- Cache clearing
- Removed multiprocessing (`sampling_processes=0`)
**Commits**: `8303063`, `184c3f6`, `acdaf52`

---

## Files Modified

### Notebook
- `D2E2S_Simple_Training.ipynb` - Main training notebook

### Code Files
- `Codebase/train.py` - Training script with debug output
- `Codebase/models/D2E2S_Model.py` - Model definition
- `Codebase/trainer/loss.py` - Loss computation with gradient accumulation

### Configuration
- `kernel-metadata.json` - Kaggle kernel settings

---

## Testing Checklist

- [ ] Verify git commit on Kaggle shows `622c703`
- [ ] Verify `hidden_dim` parameter shows as 384
- [ ] Verify LSTM output dimension is 768
- [ ] Verify training starts without dimension errors
- [ ] Monitor GPU memory usage (should be < 15GB)

---

## Resources

- **GitHub Repo**: https://github.com/VishalRepos/D2E2S
- **Kaggle Notebook**: https://www.kaggle.com/code/uom239364u/d2e2s-training
- **Latest Commit**: 622c703
