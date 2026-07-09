#!/bin/bash
# Script to set up HuggingFace authentication for cluster

echo "Setting up HuggingFace authentication..."

# Option 1: Use environment variable (recommended for cluster)
# Add this to your SLURM script or ~/.bashrc:
# export HF_TOKEN="your_huggingface_token_here"

# Option 2: Login via CLI (interactive)
# huggingface-cli login

# Option 3: Add to ~/.netrc (if you have a token)
# Add these lines to ~/.netrc:
# machine huggingface.co
# login your_username
# password your_hf_token

echo ""
echo "Choose one of the following methods:"
echo ""
echo "1. Environment Variable (Recommended for SLURM):"
echo "   Add to your SLURM script or job:"
echo "   export HF_TOKEN='hf_your_token_here'"
echo ""
echo "2. Interactive Login:"
echo "   huggingface-cli login"
echo ""
echo "3. Add to ~/.netrc (same file as wandb):"
echo "   echo 'machine huggingface.co' >> ~/.netrc"
echo "   echo 'login your_hf_username' >> ~/.netrc"
echo "   echo 'password hf_your_token_here' >> ~/.netrc"
echo "   chmod 600 ~/.netrc"
echo ""
echo "Get your HuggingFace token from: https://huggingface.co/settings/tokens"

