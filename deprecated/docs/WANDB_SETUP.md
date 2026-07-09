# Weights & Biases (wandb) Setup for LightGen

## Quick Setup

### 1. Install wandb (if not already installed)
```bash
cd /localhome/dya78/code/lightgen/TEXGen
conda activate texgen
pip install wandb
```

### 2. Login to wandb
```bash
wandb login
```

You'll be prompted for your API key. Get it from: https://wandb.ai/authorize

Or set it as environment variable:
```bash
export WANDB_API_KEY=your_api_key_here
```

### 3. Train with wandb monitoring

**For full dataset (PointUVNet):**
```bash
python launch.py --config configs/lightgen_pointuv.yaml --gpu 0 --train --wandb
```

**For full dataset (SimpleUVUNet):**
```bash
python launch.py --config configs/lightgen_full.yaml --gpu 0 --train --wandb
```

**For overfitting test:**
```bash
python launch.py --config configs/lightgen_pointuv_overfit.yaml --gpu 0 --train --wandb
```

## What Gets Logged

### Metrics (Every Step)
- `train/loss` - Total training loss
- `train/mse` - MSE loss component
- `train/l1` - L1 loss component
- `train/lr` - Learning rate

### Validation Metrics (Every Epoch)
- `val_psnr` - Peak Signal-to-Noise Ratio (higher is better)
- `val_ssim` - Structural Similarity Index (higher is better)
- `val_lpips` - Perceptual similarity (lower is better)

### Images
- Training samples (every 100 steps)
- Validation predictions (every epoch)
- Ground truth comparisons

### System Info
- GPU memory usage
- Training speed (it/s)
- Epoch time

## View Your Results

After starting training, your runs will be available at:
```
https://wandb.ai/YOUR_USERNAME/LightGen
```

## Features on wandb Dashboard

1. **Real-time metrics plotting**
   - Compare multiple runs
   - Zoom into specific training phases

2. **Image visualization**
   - See predictions evolving over training
   - Compare with ground truth

3. **Hyperparameter comparison**
   - Compare different configs side-by-side
   - See which settings work best

4. **System monitoring**
   - GPU utilization
   - Memory usage
   - Training speed

5. **Alerts**
   - Set up email alerts for training issues
   - Get notified when training completes

## Comparing Runs

### Compare SimpleUVUNet vs PointUVNet

```bash
# Terminal 1: Train SimpleUVUNet
python launch.py --config configs/lightgen_full.yaml --gpu 0 --train --wandb

# Terminal 2: Train PointUVNet (or wait for first to finish)
python launch.py --config configs/lightgen_pointuv.yaml --gpu 0 --train --wandb
```

Then compare them on wandb dashboard:
- Select both runs
- Click "Compare" 
- See side-by-side metrics

## Offline Mode (Optional)

If you want to save logs locally and sync later:

```bash
export WANDB_MODE=offline
python launch.py --config configs/lightgen_pointuv.yaml --gpu 0 --train --wandb

# Later, sync to wandb:
wandb sync outputs/lightgen/pointuv_full@TIMESTAMP/wandb/
```

## Custom Project/Team

To log to a specific team or project:

Edit `launch.py` line 160-162:
```python
wandb_logger = WandbLogger(
    project="LightGen",           # Your project name
    entity="your_team_name",      # Your team (optional)
    name=f"{cfg.name}-{cfg.tag}", # Run name
    config=dict(cfg),             # Config to log
)
```

## Useful wandb Commands

```bash
# List your runs
wandb runs LightGen

# Pull a specific run's files
wandb pull YOUR_USERNAME/LightGen/RUN_ID

# Delete a run
wandb delete YOUR_USERNAME/LightGen/RUN_ID

# Export run data
wandb export YOUR_USERNAME/LightGen/RUN_ID
```

## Troubleshooting

### "wandb: ERROR Error uploading"
- Check internet connection
- Try: `wandb login --relogin`

### "wandb: Network failure"
- Use offline mode: `export WANDB_MODE=offline`

### Too many images logged (slow)
- Images are logged every 100 steps by default
- Adjust in config: `check_train_every_n_steps: 500`

### Run names are too generic
- They're automatically set as `{cfg.name}-{cfg.tag}`
- For custom names, modify the config's `tag` field

## Example Dashboard

After training starts, you'll see:

```
wandb: 🚀 View run at https://wandb.ai/username/LightGen/runs/xyz123
wandb: Syncing run lightgen-pointuv_full
wandb: ⭐️ View project at https://wandb.ai/username/LightGen
```

The dashboard will show:
- Loss curves updating in real-time
- Validation metrics after each epoch  
- Training images as they're generated
- GPU memory/utilization graphs
- Hyperparameters and config
- Code version (git commit)

## Best Practices

1. **Use descriptive tags**: Update `tag` in config to describe the experiment
2. **Add notes**: Add run notes on wandb after starting
3. **Group related runs**: Use consistent naming for comparison
4. **Star best runs**: Mark good runs with star on wandb
5. **Add comments**: Document findings on wandb dashboard

Happy training! 🚀



