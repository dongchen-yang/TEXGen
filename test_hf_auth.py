#!/usr/bin/env python3
"""
Test script to verify HuggingFace authentication and fallback mechanism
"""

import os
import sys

print("=" * 60)
print("Testing HuggingFace Authentication")
print("=" * 60)

# Check for HF token
hf_token = os.environ.get('HF_TOKEN', None) or os.environ.get('HUGGING_FACE_HUB_TOKEN', None)
print(f"\n1. HF_TOKEN environment variable: {'✓ Set' if hf_token else '✗ Not set'}")

# Check for cached token
try:
    from huggingface_hub import HfFolder
    cached_token = HfFolder.get_token()
    print(f"2. Cached HF token (from huggingface-cli login): {'✓ Found' if cached_token else '✗ Not found'}")
except:
    print("2. Cached HF token: ✗ Cannot check")

# Test loading tokenizer
print("\n3. Testing tokenizer loading...")
try:
    from transformers import CLIPTokenizer
    
    # Try primary model
    print("   Trying stabilityai/stable-diffusion-2-depth...")
    use_auth_token = hf_token if hf_token else True
    try:
        tokenizer = CLIPTokenizer.from_pretrained(
            "stabilityai/stable-diffusion-2-depth",
            subfolder="tokenizer",
            token=use_auth_token
        )
        print("   ✓ SUCCESS: Loaded stabilityai/stable-diffusion-2-depth")
    except Exception as e:
        print(f"   ✗ FAILED: {type(e).__name__}: {str(e)[:100]}")
        
        # Try fallback
        print("\n   Trying fallback: stabilityai/stable-diffusion-2-1...")
        try:
            tokenizer = CLIPTokenizer.from_pretrained(
                "stabilityai/stable-diffusion-2-1",
                subfolder="tokenizer"
            )
            print("   ✓ SUCCESS: Loaded stabilityai/stable-diffusion-2-1 (fallback)")
        except Exception as e2:
            print(f"   ✗ FAILED: {type(e2).__name__}: {str(e2)[:100]}")
            print("\n❌ Both primary and fallback models failed!")
            sys.exit(1)

except ImportError as e:
    print(f"   ✗ FAILED: Cannot import transformers: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All checks passed! Training should work.")
print("=" * 60)

