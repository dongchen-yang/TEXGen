#!/usr/bin/env python3
"""
Create an emission strength animation by calling render_inference_blender.py multiple times.
All frames are saved under videos/{sample_id}/ folder.

Usage:
    conda activate lightgen
    python create_emission_animation.py --sample-id fff48e914c4847a08660b9e08b1b733c --frames 20
"""

import os
import subprocess
import argparse
import numpy as np
from pathlib import Path

def create_emission_animation(sample_id, num_frames=20, output_dir='videos'):
    """Create emission strength animation for a sample."""

    print(f"Creating emission strength animation for {sample_id}")
    print(f"Frames: {num_frames}, Output: {output_dir}/{sample_id}")

    # Create output directory: videos/{sample_id}/
    sample_output_dir = Path(output_dir) / sample_id
    sample_output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate emission strengths (0 to 2.0)
    strength_values = np.linspace(0.0, 2.0, num_frames)

    script_dir = Path(__file__).parent

    # Render each frame
    for i, strength in enumerate(strength_values):
        print(f"  Frame {i}/{num_frames-1}: strength {strength:.2f}")

        # Temp output dir for this frame
        temp_dir = sample_output_dir / f"temp_{i:04d}"
        
        cmd = [
            "python", "render_inference_blender.py",
            "--inference-dir", "inference_outputs",
            "--output-dir", str(temp_dir),
            "--sample-ids", sample_id,
            "--emission-strength", str(strength),
            "--resolution", "256"
        ]

        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=300,
                cwd=str(script_dir)
            )
            if result.returncode != 0:
                print(f"    Error: {result.stderr[-500:] if result.stderr else 'Unknown error'}")
            else:
                # Move the comparison.png to frame_{i}.png
                comparison_src = temp_dir / sample_id / "comparison.png"
                comparison_dst = sample_output_dir / f"frame_{i:04d}.png"
                if comparison_src.exists():
                    import shutil
                    shutil.copy2(comparison_src, comparison_dst)
                    shutil.rmtree(temp_dir)  # Clean up temp dir
                    print(f"    ✓ Saved {comparison_dst.name}")
                else:
                    print(f"    Warning: comparison.png not found")
        except subprocess.TimeoutExpired:
            print(f"    Timeout")
        except Exception as e:
            print(f"    Failed: {e}")

    # Count frames
    frame_files = sorted(sample_output_dir.glob("frame_*.png"))
    print(f"\nCollected {len(frame_files)}/{num_frames} frames")

    if len(frame_files) == 0:
        print("No frames created, skipping video")
        return False

    # Create video with ffmpeg
    video_path = sample_output_dir / f"{sample_id}_emission_animation.mp4"
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-framerate", "5",
        "-i", str(sample_output_dir / "frame_%04d.png"),
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
            print(f"FFmpeg error: {result.stderr[-300:] if result.stderr else 'Unknown'}")
            return False
    except Exception as e:
        print(f"Failed to create video: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Create emission strength animation')
    parser.add_argument('--sample-id', type=str, required=True,
                        help='Sample ID to animate')
    parser.add_argument('--frames', type=int, default=20,
                        help='Number of animation frames')
    parser.add_argument('--output-dir', type=str, default='videos',
                        help='Output directory for video')

    args = parser.parse_args()

    success = create_emission_animation(args.sample_id, args.frames, args.output_dir)

    if success:
        print("✓ Animation completed!")
    else:
        print("✗ Animation failed")

if __name__ == "__main__":
    main()