# LightGen Resolution Configurations

This directory contains multiple configuration files for training LightGen at different resolutions.

## Available Configs

### 1. `lightgen_pointuv.yaml` (512x512) - **Default**
- **Resolution**: 512×512 UV maps
- **Batch Size**: 2
- **Use Case**: High-quality training, final models
- **Memory**: ~11GB per GPU (approximate)
- **Training Speed**: Slower, but better quality
- **Output Tag**: `pointuv_full`

### 2. `lightgen_pointuv_256.yaml` (256x256) - **Fast Training**
- **Resolution**: 256×256 UV maps  
- **Batch Size**: 8 (4x larger)
- **Use Case**: Fast prototyping, hyperparameter tuning, debugging
- **Memory**: ~11GB per GPU (approximate, same as 512 due to larger batch)
- **Training Speed**: 4x faster per epoch (lower resolution + larger batch)
- **Output Tag**: `pointuv_256res`

### 3. `lightgen_overfit_10.yaml` - **Overfitting Test**
- **Resolution**: 512×512
- **Batch Size**: 2
- **Use Case**: Quick sanity check, verify model can learn
- **Data**: Only 10 samples (train=val=test)
- **Output Tag**: `overfit_10_samples`

## Key Differences (256 vs 512)

| Parameter | 512 Config | 256 Config | Notes |
|-----------|------------|------------|-------|
| `uv_height/uv_width` | 512 | 256 | UV map resolution |
| `height/width` | 512 | 256 | Rendered view resolution |
| `batch_size` | 2 | 8 | 4x larger batch |
| `window_size` | [0, 256, 256, 512, 1024] | [0, 128, 128, 256, 256] | Attention window sizes |
| `lr` | 1e-4 | 1.5e-4 | Slightly higher for larger batch |

## When to Use Each

### Use 512x512 (`lightgen_pointuv.yaml`) when:
- ✅ Training your final production model
- ✅ You need high-quality outputs
- ✅ You have plenty of training time
- ✅ You want maximum detail in emission maps

### Use 256x256 (`lightgen_pointuv_256.yaml`) when:
- ✅ Prototyping new features or architectures
- ✅ Tuning hyperparameters (LR, dropout, etc.)
- ✅ Quick experiments to test ideas
- ✅ Debugging data pipeline or training code
- ✅ Limited compute time or want faster iterations
- ✅ Initial model development before scaling up

### Use Overfit Test (`lightgen_overfit_10.yaml`) when:
- ✅ Verifying model can learn at all
- ✅ Debugging training bugs
- ✅ Testing new loss functions
- ✅ Quick sanity checks (< 10 minutes)

## Launch Commands

### 512x512 Training:
```bash
python launch.py --config configs/lightgen_pointuv.yaml --gpu 0
```

### 256x256 Training:
```bash
python launch.py --config configs/lightgen_pointuv_256.yaml --gpu 0
```

### Overfit Test:
```bash
python launch.py --config configs/lightgen_overfit_10.yaml --gpu 0
```

## Expected Training Times (Approximate)

Based on a single GPU (e.g., RTX 4090 or A100):

| Config | Samples/sec | Time per Epoch | Time to 100 Epochs |
|--------|-------------|----------------|-------------------|
| 512x512 (batch=2) | ~2-3 | ~4 hours | ~17 days |
| 256x256 (batch=8) | ~10-12 | ~1 hour | ~4 days |
| Overfit (10 samples) | N/A | ~30 sec | ~10 min (200 epochs) |

*Note: Times are approximate and depend on GPU, dataset size, and system configuration.*

## Memory Optimization Tips

If you run out of memory even with 256x256:
1. Reduce `batch_size` further (e.g., 4 or 6)
2. Use `precision: "16-mixed"` instead of `bf16-mixed`
3. Enable gradient checkpointing (if implemented)
4. Reduce `block_out_channels` (architecture modification)

## Quality Comparison

**256x256 models** can still produce good results but may have:
- Slightly less detail in fine textures
- Some loss of high-frequency information
- Still suitable for most applications

**Recommendation**: Train on 256x256 for rapid iteration, then train your best model on 512x512 for final deployment.




