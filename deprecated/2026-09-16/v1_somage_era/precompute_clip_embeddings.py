"""
Precompute CLIP image embeddings for all thumbnails to avoid running
the CLIP encoder during training. Also precomputes the fixed text
embedding for the "emission generation" prompt.

Usage:
    python precompute_clip_embeddings.py --data_root ../data/baked_uv_local_subset

Outputs:
    {data_root}/clip_embeddings/
        {sample_id}.pt          — image embedding [768]
        _text_embedding.pt      — text embedding [768] for "emission generation"
"""

import argparse
import os
from glob import glob

import torch
from PIL import Image
import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Precompute CLIP embeddings for thumbnails")
    parser.add_argument("--data_root", type=str, default="../data/baked_uv_local_subset",
                        help="Path to data root containing thumbnails/ directory")
    parser.add_argument("--clip_model", type=str, default="lambdalabs/sd-image-variations-diffusers",
                        help="CLIP model path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"])
    args = parser.parse_args()

    weight_dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)

    # Load CLIP image encoder
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    print(f"Loading CLIP image encoder from {args.clip_model}...")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.clip_model, subfolder="image_encoder"
    ).to(device, dtype=weight_dtype)
    image_encoder.eval()

    feature_extractor = CLIPImageProcessor.from_pretrained(
        args.clip_model, subfolder="feature_extractor"
    )

    clip_image_mean = torch.as_tensor(feature_extractor.image_mean)[:, None, None].to(device, dtype=weight_dtype)
    clip_image_std = torch.as_tensor(feature_extractor.image_std)[:, None, None].to(device, dtype=weight_dtype)
    crop_h = feature_extractor.crop_size['height']
    crop_w = feature_extractor.crop_size['width']

    # Load CLIP text encoder for the fixed prompt
    from transformers import CLIPTextModel, CLIPTokenizer
    print("Loading CLIP text encoder from stabilityai/stable-diffusion-3.5-large...")
    text_tokenizer = CLIPTokenizer.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large", subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large", subfolder="text_encoder"
    ).to(device, dtype=weight_dtype)
    text_encoder.eval()

    # Output directory
    out_dir = os.path.join(args.data_root, "clip_embeddings")
    os.makedirs(out_dir, exist_ok=True)

    # Precompute text embedding for fixed prompt
    prompt = "emission generation"
    with torch.no_grad():
        prompt_ids = text_tokenizer(
            [prompt], max_length=text_tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids.to(device)
        text_embedding = text_encoder(prompt_ids)[1].squeeze(0).cpu()  # [768]

    text_path = os.path.join(out_dir, "_text_embedding.pt")
    torch.save(text_embedding, text_path)
    print(f"Saved text embedding to {text_path} (shape: {text_embedding.shape})")

    # Precompute image embeddings for all thumbnails
    thumbnail_dir = os.path.join(args.data_root, "thumbnails")
    thumbnail_files = sorted(glob(os.path.join(thumbnail_dir, "*.png")))
    print(f"Found {len(thumbnail_files)} thumbnails in {thumbnail_dir}")

    from torchvision.transforms.functional import resize
    from torchvision.transforms import InterpolationMode

    skipped = 0
    for thumb_path in tqdm(thumbnail_files, desc="Encoding thumbnails"):
        sample_id = os.path.splitext(os.path.basename(thumb_path))[0]
        out_path = os.path.join(out_dir, f"{sample_id}.pt")

        # Skip if already computed
        if os.path.exists(out_path):
            skipped += 1
            continue

        # Load and preprocess thumbnail (same as ClipTokenizer.process_image)
        img = Image.open(thumb_path).convert('RGB')
        img = img.resize((224, 224), Image.BILINEAR)
        img_tensor = torch.from_numpy(np.array(img)).float() / 255.0  # [224, 224, 3]
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device, dtype=weight_dtype)  # [1, 3, 224, 224]

        # CLIP preprocessing: normalize and resize to crop size
        img_tensor = (img_tensor - clip_image_mean) / clip_image_std
        img_tensor = resize(img_tensor, (crop_h, crop_w), interpolation=InterpolationMode.BICUBIC)

        with torch.no_grad():
            image_embedding = image_encoder(img_tensor).image_embeds.squeeze(0).cpu()  # [768]

        torch.save(image_embedding, out_path)

    print(f"Done! {len(thumbnail_files) - skipped} new, {skipped} skipped (already existed)")
    print(f"Embeddings saved to {out_dir}")

    # Cleanup
    del image_encoder, text_encoder
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
