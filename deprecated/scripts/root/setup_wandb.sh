#!/bin/bash
# Setup script for Weights & Biases monitoring

echo "=================================="
echo "Setting up Weights & Biases (wandb)"
echo "=================================="

# Check if wandb is installed
if ! conda list | grep -q wandb; then
    echo "Installing wandb..."
    pip install wandb -q
else
    echo "✓ wandb is already installed"
fi

echo ""
echo "Now you need to login to wandb:"
echo "1. Get your API key from: https://wandb.ai/authorize"
echo "2. Run: wandb login"
echo ""
echo "Or login now:"
read -p "Do you want to login now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    wandb login
fi

echo ""
echo "=================================="
echo "✓ Setup complete!"
echo "=================================="
echo ""
echo "To train with wandb monitoring:"
echo "  python launch.py --config configs/lightgen_pointuv.yaml --gpu 0 --train --wandb"
echo ""
echo "Your runs will be logged to: https://wandb.ai/YOUR_USERNAME/LightGen"



