#!/bin/bash
# Script to copy HuggingFace cache to cluster

echo "Copying HuggingFace cache for stable-diffusion-2-depth to cluster..."

# Source directory (local cache)
SRC="$HOME/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-depth"

# Target cluster (UPDATE THIS with your cluster address)
CLUSTER_USER="your_username"  # Update this
CLUSTER_HOST="your_cluster_address"  # Update this
CLUSTER_CACHE="~/.cache/huggingface/hub/"

echo "Source: $SRC"
echo "Target: ${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_CACHE}"
echo ""

# Check if source exists
if [ ! -d "$SRC" ]; then
    echo "Error: Source directory not found: $SRC"
    exit 1
fi

# Create target directory on cluster
echo "Creating target directory on cluster..."
ssh ${CLUSTER_USER}@${CLUSTER_HOST} "mkdir -p ${CLUSTER_CACHE}"

# Copy the entire model cache
echo "Copying model cache (this may take a few minutes)..."
rsync -avz --progress "$SRC" ${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_CACHE}

echo ""
echo "✓ Done! The model cache has been copied to the cluster."
echo "You can now run training with HF_HUB_OFFLINE=1 to use cached models only."

