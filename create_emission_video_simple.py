#!/usr/bin/env python3
"""
Create an emission strength animation video (simplified version).

This script creates individual Blender scripts for each frame and runs them.
"""

import os
import subprocess
import argparse
import numpy as np
from pathlib import Path
import pandas as pd

def get_glb_path(sample_id):
    """Get GLB path for a sample."""
    df = pd.read_parquet('/localhome/dya78/code/lightgen/data/baked_uv_local/df_SomgProc_filtered.parquet')
    glb_path = df.loc[sample_id, 'glb_1k_path']
    full_glb_path = Path('/cs/3dlg-falas/datasets/TexVerse-1K') / glb_path
    return str(full_glb_path)

def create_blender_frame_script(frame_path, glb_path, emission_path, emission_strength):
    """Create a Blender script that renders one frame."""

    script_content = f'''
import bpy
import sys
import numpy as np
from pathlib import Path
from PIL import Image

# Add xgutils to path
sys.path.append("/local-scratch/localhome/dya78/code/lightgen/external/xgutils")
from xgutils import bpyutil

def apply_emission_texture_to_materials(obj, emission_img_blender, emission_strength=1.0):
    """Apply emission texture to all materials of an object."""
    for mat in obj.data.materials:
        if mat is None:
            continue

        # Create nodes if not already present
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # Clear existing nodes
        nodes.clear()

        # Create Principled BSDF
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        bsdf.location = (0, 0)

        # Create Emission node
        emission = nodes.new('ShaderNodeEmission')
        emission.location = (-300, 0)

        # Create UV Map node
        uv_node = nodes.new('ShaderNodeUVMap')
        uv_node.location = (-600, 0)

        # Create Image Texture node
        tex_node = nodes.new('ShaderNodeTexImage')
        tex_node.location = (-450, 0)
        tex_node.image = emission_img_blender

        # Create Material Output
        output = nodes.new('ShaderNodeOutputMaterial')
        output.location = (300, 0)

        # Link nodes
        links.new(uv_node.outputs['UV'], tex_node.inputs['Vector'])
        links.new(tex_node.outputs['Color'], emission.inputs['Color'])
        links.new(emission.outputs['Emission'], bsdf.inputs['Emission'])
        links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

        # Set emission strength
        emission.inputs['Strength'].default_value = {emission_strength}

print("Starting Blender rendering...")

glb_path = "{glb_path}"
emission_path = "{emission_path}"

print(f"GLB path: {{glb_path}}")
print(f"Emission path: {{emission_path}}")

if not Path(glb_path).exists():
    print(f"GLB not found: {{glb_path}}")
    sys.exit(1)

if not Path(emission_path).exists():
    print(f"Emission texture not found: {{emission_path}}")
    sys.exit(1)

# Load emission texture
emission_img = np.array(Image.open(emission_path)).astype(np.float32) / 255.0
print(f"Emission image shape: {{emission_img.shape}}")

# Load GLB
print("Loading GLB...")
obj = bpyutil.load_glb(glb_path, import_shading=None)

# Create emission texture in Blender
tex_h, tex_w = emission_img.shape[:2]
emission_img_blender = bpy.data.images.new("EmissionTexture", width=tex_w, height=tex_h, alpha=False)

# Convert to Blender format
emission_rgba = np.ones((tex_h, tex_w, 4), dtype=np.float32)
emission_rgba[:, :, :3] = emission_img.astype(np.float32)
emission_rgba = np.flipud(emission_rgba)
emission_img_blender.pixels = emission_rgba.flatten().tolist()

# Apply emission texture
print(f"Applying emission texture with strength {emission_strength}...")
apply_emission_texture_to_materials(obj, emission_img_blender, {emission_strength})

# Set camera (front view)
bpyutil.set_camera_pos((2, 0, 0))

# Render settings
bpy.context.scene.render.filepath = "{frame_path}"
bpy.context.scene.render.resolution_x = 512
bpy.context.scene.render.resolution_y = 512
bpy.context.scene.cycles.samples = 64

print(f"Rendering to: {frame_path}")
bpy.ops.render.render(write_still=True)
print("Render completed!")
'''

    return script_content

def create_emission_video(sample_id, inference_dir, output_dir, strength_range=(0, 2.0), num_frames=20):
    """Create a video showing emission strength animation."""

    print(f"Creating emission strength animation for sample {sample_id}")

    # Get GLB path and emission path
    try:
        glb_path = get_glb_path(sample_id)
        emission_path = str(Path(inference_dir) / sample_id / "gt_emission.png")
        print(f"GLB path: {glb_path}")
        print(f"Emission path: {emission_path}")
    except Exception as e:
        print(f"Error getting paths: {e}")
        return False

    if not Path(glb_path).exists():
        print(f"GLB not found: {glb_path}")
        return False

    if not Path(emission_path).exists():
        print(f"Emission texture not found: {emission_path}")
        return False

    # Create frames directory
    frames_dir = output_dir / f"{sample_id}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Calculate strength values
    min_strength, max_strength = strength_range
    strength_values = np.linspace(min_strength, max_strength, num_frames)

    # Render each frame
    blender_exe = "/localhome/dya78/software/blender-3.2.0-linux-x64/blender"

    for i, strength in enumerate(strength_values):
        frame_path = frames_dir / f"frame_{i:04d}.png"
        script_path = frames_dir / f"script_{i:04d}.py"

        print(f"Rendering frame {i}/{num_frames-1}: emission strength {strength:.1f}")
        # Create Blender script for this frame
        script_content = create_blender_frame_script(
            str(frame_path), glb_path, emission_path, strength
        )

        with open(script_path, 'w') as f:
            f.write(script_content)

        # Run Blender
        cmd = [blender_exe, "--background", "--python", str(script_path)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                  cwd=str(Path(__file__).parent), timeout=120)
            if result.returncode != 0:
                print(f"Error rendering frame {i}: {result.stderr[-1000:]}")  # Last 1000 chars
                print(f"Return code: {result.returncode}")
            else:
                print(f"✓ Frame {i} completed")
        except subprocess.TimeoutExpired:
            print(f"Timeout rendering frame {i}")
        except Exception as e:
            print(f"Failed to render frame {i}: {e}")

    # Check how many frames were created
    frame_files = list(frames_dir.glob("frame_*.png"))
    print(f"Created {len(frame_files)} frames out of {num_frames}")

    if len(frame_files) == 0:
        print("No frames created, aborting video creation")
        return False

    # Create video using ffmpeg
    video_path = output_dir / f"{sample_id}_emission_animation.mp4"
    frames_pattern = str(frames_dir / "frame_%04d.png")

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-framerate", "5",  # Slower framerate for visibility
        "-i", frames_pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(video_path)
    ]

    print("Creating video...")
    try:
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Video created: {video_path}")
            return True
        else:
            print(f"FFmpeg error: {result.stderr[-500:]}")
            return False
    except Exception as e:
        print(f"Failed to create video: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Create emission strength animation video')
    parser.add_argument('--sample-id', type=str, required=True,
                        help='Sample ID to create animation for')
    parser.add_argument('--inference-dir', type=str, default='inference_outputs',
                        help='Directory containing inference outputs')
    parser.add_argument('--output-dir', type=str, default='videos',
                        help='Directory to save video')
    parser.add_argument('--min-strength', type=float, default=0.0,
                        help='Minimum emission strength')
    parser.add_argument('--max-strength', type=float, default=2.0,
                        help='Maximum emission strength')
    parser.add_argument('--num-frames', type=int, default=20,
                        help='Number of frames in animation')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    success = create_emission_video(
        sample_id=args.sample_id,
        inference_dir=args.inference_dir,
        output_dir=output_dir,
        strength_range=(args.min_strength, args.max_strength),
        num_frames=args.num_frames
    )

    if success:
        print("✓ Animation video created successfully!")
    else:
        print("✗ Failed to create animation video")

if __name__ == "__main__":
    main()