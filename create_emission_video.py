#!/usr/bin/env python3
"""
Create an emission strength animation video.

This script renders multiple frames with varying emission strength (0 to 2.0)
and combines them into a video using ffmpeg.
"""

import os
import subprocess
import argparse
import numpy as np
from pathlib import Path

def create_emission_video(sample_id, inference_dir, output_dir, strength_range=(0, 2.0), num_frames=30):
    """
    Create a video showing emission strength animation for a specific sample.

    Args:
        sample_id: Sample ID to render
        inference_dir: Directory with inference outputs
        output_dir: Directory to save video
        strength_range: Tuple of (min_strength, max_strength)
        num_frames: Number of frames in the animation
    """
    print(f"Creating emission strength animation for sample {sample_id}")
    print(f"Strength range: {strength_range[0]} to {strength_range[1]}")
    print(f"Number of frames: {num_frames}")

    # Create temporary directory for frames
    frames_dir = output_dir / f"{sample_id}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Calculate strength values for each frame
    min_strength, max_strength = strength_range
    strength_values = np.linspace(min_strength, max_strength, num_frames)

    # Get paths
    inference_dir = Path(inference_dir)
    sample_infer_dir = inference_dir / sample_id

    # Check if emission textures exist
    gt_emission_path = sample_infer_dir / "gt_emission.png"
    pred_emission_path = sample_infer_dir / "pred_emission.png"

    if not gt_emission_path.exists() or not pred_emission_path.exists():
        print(f"Missing emission textures for {sample_id}")
        return False

    print(f"Found emission textures at {sample_infer_dir}")

    # Render each frame
    for i, strength in enumerate(strength_values):
        frame_num = str(i).zfill(4)
        frame_path = frames_dir / f"frame_{frame_num}.png"

        print(f"Rendering frame {i}/{num_frames-1}: emission strength {strength:.1f}")
        # Run Blender rendering with specific emission strength
        blender_exe = "/localhome/dya78/software/blender-3.2.0-linux-x64/blender"
        cmd = [
            blender_exe, "--background",
            "--python", "render_inference_blender.py",
            "--inference-dir", str(inference_dir),
            "--output-dir", str(frames_dir / f"temp_{frame_num}"),
            "--sample-ids", sample_id,
            "--emission-strength", str(strength),
            "--resolution", "256"  # Smaller resolution for video
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(__file__).parent))
            if result.returncode != 0:
                print(f"Error rendering frame {i}: {result.stderr}")
                continue
        except Exception as e:
            print(f"Failed to render frame {i}: {e}")
            continue

        # Find the rendered comparison image and copy it as the frame
        temp_output_dir = frames_dir / f"temp_{frame_num}"
        comparison_path = temp_output_dir / sample_id / "comparison.png"

        if comparison_path.exists():
            # Copy comparison image as frame
            import shutil
            shutil.copy2(comparison_path, frame_path)
        else:
            print(f"Comparison image not found for frame {i}")

        # Clean up temporary directory
        import shutil
        if temp_output_dir.exists():
            shutil.rmtree(temp_output_dir)

    # Create video using ffmpeg
    video_path = output_dir / f"{sample_id}_emission_animation.mp4"
    frames_pattern = str(frames_dir / "frame_%04d.png")

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-framerate", "10",  # 10 fps
        "-i", frames_pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # Ensure even dimensions
        str(video_path)
    ]

    print(f"Creating video: {' '.join(ffmpeg_cmd)}")
    try:
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Video created: {video_path}")
            return True
        else:
            print(f"Error creating video: {result.stderr}")
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
    parser.add_argument('--num-frames', type=int, default=30,
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