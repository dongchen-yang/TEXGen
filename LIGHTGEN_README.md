# LightGen: Emission Map Generation

This is a modified version of TEXGen for generating emission maps from geometry and material properties.

## Overview

**Input:**
- Geometry: 3D position map and normal map in UV space
- Material: Albedo (color), roughness, and metallic maps

**Output:**
- Emission map in UV space (RGB values indicating light emission)

## Data Format

The data should be organized in the `/localhome/dya78/code/lightgen/data/baked_uv` directory with the following structure:

```
data/baked_uv/
├── 000-000/
│   ├── {sample_id}/
│   │   ├── somage.npz      # Contains all UV maps
│   │   ├── somage.png      # Visualization (optional)
│   │   └── _dproc_*.json   # Metadata (optional)
│   └── ...
├── 000-001/
│   └── ...
└── ...
```

### NPZ File Format

Each `somage.npz` file should contain:

- `occupancy`: [512, 512, 1] bool - mask indicating valid UV coordinates
- `position`: [512, 512, 3] uint16 - 3D positions (decoded to float range [-2, 2])
- `objnormal`: [512, 512, 3] uint16 - object-space normals (decoded to float range [-1, 1])
- `color`: [512, 512, 3] uint8 - albedo/diffuse color (decoded to float range [0, 1])
- `metal`: [512, 512, 1] uint8 - metallic values (decoded to float range [0, 1])
- `rough`: [512, 512, 1] uint8 - roughness values (decoded to float range [0, 1])
- `emission_color`: [512, 512, 3] uint8 - target emission map (decoded to float range [0, 1])

## Model Architecture

The model uses a U-Net style architecture with:
- **Input channels**: 11
  - Position map: 3 channels
  - Normal map: 3 channels
  - Albedo map: 3 channels
  - Metallic map: 1 channel
  - Roughness map: 1 channel
- **Output channels**: 3 (RGB emission map)

The model is based on diffusion models with the following key components:
- Multi-scale UV-aware processing
- Point-UV attention blocks
- Diffusion-based generation with v-prediction
- Classifier-free guidance for better control

## Training

### Quick Start

1. **Verify data structure:**
```bash
cd /localhome/dya78/code/lightgen
python3 -c "
import numpy as np
import os
data_root = 'data/baked_uv'
samples = []
for subdir in sorted(os.listdir(data_root)):
    subdir_path = os.path.join(data_root, subdir)
    if os.path.isdir(subdir_path):
        for sample_id in sorted(os.listdir(subdir_path)):
            npz_file = os.path.join(subdir_path, sample_id, 'somage.npz')
            if os.path.exists(npz_file):
                samples.append(sample_id)
print(f'Found {len(samples)} samples')
print(f'First 5 samples: {samples[:5]}')
"
```

2. **Update data indices in config:**
Edit `configs/lightgen.yaml` and adjust the train/val/test split based on your dataset size:
```yaml
data:
  train_indices: [0, 800]   # Adjust based on your dataset
  val_indices: [800, 900]
  test_indices: [900, 1000]
```

3. **Train the model:**
```bash
cd TEXGen
python launch.py --config configs/lightgen.yaml --train --gpu 0
```

For multi-GPU training:
```bash
python launch.py --config configs/lightgen.yaml --train --gpu 0,1,2,3
```

### Training Options

- `--train`: Start training
- `--validate`: Run validation only
- `--test`: Run testing only
- `--export`: Export results
- `--wandb`: Log to Weights & Biases
- `--verbose`: Enable debug logging
- `--gpu`: Specify GPU IDs (e.g., "0" or "0,1,2,3")

### Resume Training

To resume from a checkpoint:
```bash
python launch.py --config configs/lightgen.yaml --train --gpu 0 \
  system.resume=/path/to/checkpoint.ckpt
```

## Testing/Inference

Run inference on the test set:
```bash
python launch.py --config configs/lightgen.yaml --test --gpu 0 \
  system.resume=/path/to/checkpoint.ckpt
```

Results will be saved in `outputs/lightgen/v1/`.

## Key Configuration Parameters

### Model Parameters

- `in_channels`: Number of input channels (11 for position + normal + albedo + metal + rough)
- `out_channels`: Number of output channels (3 for RGB emission)
- `block_out_channels`: Hidden dimensions at each scale
- `num_layers`: Number of layers at each scale
- `use_uv_head`: Enable UV-specific processing head

### Diffusion Parameters

- `prediction_type`: Type of prediction ("v_prediction", "epsilon", or "sample")
- `test_num_steps`: Number of denoising steps during inference (50 recommended)
- `test_cfg_scale`: Classifier-free guidance scale (2.0 recommended)
- `condition_drop_rate`: Probability of dropping condition during training (0.1 recommended)

### Training Parameters

- `batch_size`: Batch size for training (4-8 recommended depending on GPU memory)
- `max_epochs`: Total number of training epochs
- `lr`: Learning rate (1e-4 recommended)
- `gradient_clip_val`: Gradient clipping value

## File Structure

```
TEXGen/
├── configs/
│   └── lightgen.yaml           # LightGen configuration
├── spuv/
│   ├── data/
│   │   └── lightgen_uv.py     # LightGen data loader
│   ├── systems/
│   │   └── lightgen_system.py # LightGen system implementation
│   └── ...
├── launch.py                   # Main training/testing script
└── LIGHTGEN_README.md          # This file
```

## Troubleshooting

### Out of Memory (OOM)

- Reduce `batch_size` in config
- Reduce `block_out_channels` to [32, 128, 256, 512, 1024]
- Use gradient checkpointing (if implemented)

### Poor Results

- Increase `test_cfg_scale` (try 3.0 or 4.0)
- Increase `test_num_steps` (try 100)
- Train for more epochs
- Check if input data is properly normalized

### Data Loading Errors

- Verify NPZ files have correct keys: `occupancy`, `position`, `objnormal`, `color`, `metal`, `rough`, `emission_color`
- Check data types match expected format
- Ensure all maps are 512x512 resolution

## Differences from Original TEXGen

1. **Data Loader**: Custom `LightGenDataset` reads from NPZ files instead of OBJ+PNG
2. **Input Format**: Takes geometry + material properties (11 channels) instead of mesh + image
3. **Output**: Generates emission maps instead of texture maps
4. **Conditioning**: Uses material properties (albedo) as visual conditioning
5. **Training**: Directly operates in UV space without 3D rendering loop

## Next Steps

- Fine-tune hyperparameters based on your specific dataset
- Add additional conditioning (e.g., environment maps, light positions)
- Experiment with different backbone architectures
- Add data augmentation if needed
- Implement custom loss functions for physically-based constraints

## Citation

If you use this code, please cite the original TEXGen paper and acknowledge the LightGen modifications.

