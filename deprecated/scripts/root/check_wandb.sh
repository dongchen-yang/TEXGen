#!/bin/bash
# Quick script to check wandb setup

echo "========================================"
echo "Checking Wandb Setup"
echo "========================================"

# Check if netrc exists
if [ -f ~/.netrc ]; then
    echo "✓ ~/.netrc exists"
    
    # Check permissions
    PERMS=$(stat -c "%a" ~/.netrc 2>/dev/null || stat -f "%A" ~/.netrc 2>/dev/null)
    if [ "$PERMS" = "600" ]; then
        echo "✓ ~/.netrc has correct permissions (600)"
    else
        echo "⚠️  ~/.netrc permissions: $PERMS (should be 600)"
        echo "   Fix with: chmod 600 ~/.netrc"
    fi
    
    # Check if it contains wandb entry
    if grep -q "api.wandb.ai" ~/.netrc; then
        echo "✓ ~/.netrc contains wandb.ai entry"
    else
        echo "✗ ~/.netrc does not contain wandb.ai entry"
    fi
else
    echo "✗ ~/.netrc not found"
fi

echo ""

# Check if wandb is installed
if command -v wandb &> /dev/null; then
    echo "✓ wandb is installed"
    echo "  Version: $(wandb --version)"
else
    echo "✗ wandb is not installed"
    echo "  Install with: pip install wandb"
    exit 1
fi

echo ""

# Try to verify wandb connection
echo "Verifying wandb connection..."
if wandb verify &> /dev/null; then
    echo "✓ Successfully authenticated with wandb"
    wandb verify 2>/dev/null | grep "Logged in"
else
    echo "✗ Failed to authenticate with wandb"
    echo ""
    echo "Try one of these:"
    echo "  1. Copy your netrc: scp ~/.netrc user@cluster:~/"
    echo "  2. Set API key: export WANDB_API_KEY='your_key'"
    echo "  3. Run: wandb login"
fi

echo ""
echo "========================================"
echo "Setup check complete!"
echo "========================================"



