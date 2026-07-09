# Training with TensorBoard Visualization

## Setup Complete! ✓

**Sample:** `416f4870df6449dfaf9533be8aa18701`
- Location: `000-081/416f4870df6449dfaf9533be8aa18701`
- Mask coverage: 41.7%
- Non-zero emission pixels: 70,984
- Emission range: [0, 206] (good signal!)

## Step 1: Start Training

Open a terminal and run:

```bash
conda activate texgen
cd /localhome/dya78/code/lightgen/TEXGen
python launch.py --config configs/lightgen.yaml --train --gpu 0
```

Training will save outputs to: `outputs/lightgen/v1/`

## Step 2: Monitor with TensorBoard

**Option A: In a new terminal (recommended)**

Open a second terminal and run:

```bash
conda activate texgen
cd /localhome/dya78/code/lightgen/TEXGen
tensorboard --logdir outputs/lightgen/v1/tb_logs --port 6006
```

Then open in browser: `http://localhost:6006`

**Option B: Background TensorBoard**

```bash
conda activate texgen
cd /localhome/dya78/code/lightgen/TEXGen
tensorboard --logdir outputs/lightgen/v1/tb_logs --port 6006 &
```

## What to Watch in TensorBoard

### 📊 Scalars Tab
- **`train/loss`**: Should decrease steadily (target: < 0.01 for overfitting)
- **`train/mse`**: Mean squared error (should decrease)
- **`train/l1`**: L1 loss (should decrease)
- **`val_psnr`**: Peak Signal-to-Noise Ratio (should increase, target: > 30 dB)
- **`val_ssim`**: Structural similarity (should increase to ~1.0)
- **`val_lpips`**: Perceptual loss (should decrease to ~0.0)

### 🖼️ Images Tab
Training will save images every 10 steps (`check_train_every_n_steps: 10`):
- **Generated emission maps** vs **Ground truth**
- **Rendered views** (if applicable)

Images are also saved to: `outputs/lightgen/v1/save/`

## Expected Overfitting Behavior

With 1 sample, you should see:

**Epoch 1-10:**
- Loss drops from ~1.0 to ~0.1
- PSNR increases to ~20 dB

**Epoch 10-50:**
- Loss drops to ~0.01
- PSNR increases to ~30 dB
- Images start looking very similar to ground truth

**Epoch 50-100:**
- Loss drops to < 0.001
- PSNR increases to > 35 dB
- Near-perfect reconstruction

## Training Configuration

```yaml
Sample: 416f4870df6449dfaf9533be8aa18701 (index 787608)
Batch size: 1
Epochs: 100
Learning rate: 5e-4
Optimizer: AdamW (no weight decay)
Prediction type: v_prediction
Test steps: 50 denoising steps
No dropout, No EMA (for faster overfitting)
```

## Useful Commands

### Stop training
Press `Ctrl+C` in the training terminal

### Resume from checkpoint
```bash
python launch.py --config configs/lightgen.yaml --train --gpu 0 \
  system.resume=outputs/lightgen/v1/ckpts/last.ckpt
```

### Run inference only
```bash
python launch.py --config configs/lightgen.yaml --test --gpu 0 \
  system.resume=outputs/lightgen/v1/ckpts/epoch=099.ckpt
```

### Check saved images
```bash
ls -lh outputs/lightgen/v1/save/
```

### View specific training image
```bash
# Images are saved as:
# outputs/lightgen/v1/save/it{step}-train.jpg
# outputs/lightgen/v1/save/it{step}-train-render.jpg
```

## Troubleshooting

### "CUDA out of memory"
The model is already optimized for 1 sample. If you still get OOM:
- Check GPU memory: `nvidia-smi`
- Reduce model size in config: decrease `block_out_channels`

### Loss not decreasing
- Check TensorBoard - loss should decrease within first 10 epochs
- If stuck at ~1.0, there might be a bug (let me know!)

### TensorBoard not showing data
- Make sure training has started (takes ~1 min to initialize)
- Refresh browser
- Check logs are being written: `ls outputs/lightgen/v1/tb_logs/`

## Next Steps After Overfitting

Once you verify overfitting works (loss < 0.01, PSNR > 30):

1. **Test on 10 samples** - verify generalization
2. **Scale to full dataset** - train on all 824k samples
3. **Tune hyperparameters** - learning rate, model size, etc.

## File Structure

```
outputs/lightgen/v1/
├── ckpts/                      # Model checkpoints
│   ├── last.ckpt              # Latest checkpoint
│   └── epoch=N.ckpt           # Best checkpoints
├── tb_logs/                    # TensorBoard logs
│   └── version_0/             # Training run
│       └── events.out.tfevents.*
├── save/                       # Saved images
│   └── it{step}-train.jpg     # Training visualizations
├── configs/                    # Config snapshot
└── cmd.txt                     # Command used for training
```

---

**Ready to start!** Just run the training command and open TensorBoard! 🚀

