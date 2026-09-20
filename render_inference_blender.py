#!/usr/bin/env python3
"""
Render inference results with GT and predicted emission textures using Blender.
Uses xgutils/bpyutil for Blender rendering.

Usage:
    python render_inference_blender.py --inference-dir inference_outputs --output-dir render_outputs
"""
import sys
import os

# Add external directory to Python path so xgutils can be imported
script_dir = os.path.dirname(os.path.abspath(__file__))
external_path = os.path.abspath(os.path.join(script_dir, '..', 'external'))
sys.path.insert(0, external_path)

from xgutils import bpyutil
from xgutils.vis import visutil
from xgutils.miscutil import preset_glb
import bpy
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd
import argparse
import math


def apply_emission_texture_to_materials(obj, emission_img_blender, emission_strength=1.0):
    """Apply emission texture to all materials of an object."""
    for mat in obj.data.materials:
        if mat is None:
            continue
        
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        
        # Find Principled BSDF node
        bsdf_node = None
        for node in nodes:
            if node.type == 'BSDF_PRINCIPLED':
                bsdf_node = node
                break
        
        if bsdf_node is None:
            # Create one if not found
            bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')
            output_node = None
            for node in nodes:
                if node.type == 'OUTPUT_MATERIAL':
                    output_node = node
                    break
            if output_node:
                links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])
        
        # Create texture node for emission
        tex_node = nodes.new(type='ShaderNodeTexImage')
        tex_node.image = emission_img_blender
        tex_node.interpolation = 'Linear'
        
        # Create UV map node
        uv_node = nodes.new(type='ShaderNodeUVMap')
        if 'UVMap' in obj.data.uv_layers:
            uv_node.uv_map = 'UVMap'
        elif len(obj.data.uv_layers) > 0:
            uv_node.uv_map = obj.data.uv_layers[0].name
        
        # Link UV to texture
        links.new(uv_node.outputs['UV'], tex_node.inputs['Vector'])
        
        # Link texture to emission
        links.new(tex_node.outputs['Color'], bsdf_node.inputs['Emission Color'])
        
        # Set emission strength
        bsdf_node.inputs['Emission Strength'].default_value = emission_strength


def render_with_emission(
    glb_path: str,
    emission_texture: np.ndarray,
    output_path: str,
    camera_position: tuple = (2, 2, 2),
    camera_up: tuple = (0, 0, 1),
    resolution: tuple = (512, 512),
    samples: int = 64,
    preset_blend: str = None,
    emission_strength: float = 1.0,
    dark_background: bool = True
):
    """
    Render a GLB with a custom emission texture applied.
    
    Args:
        glb_path: Path to the GLB file
        emission_texture: Emission texture as numpy array [H, W, 3] in [0, 1]
        output_path: Path to save the rendered image
        camera_position: Camera position (x, y, z)
        camera_up: Camera up vector
        resolution: Render resolution (width, height)
        samples: Number of render samples
        preset_blend: Path to preset blend file
        dark_background: If True, use dark background to show emission better
    """
    # Get default preset if not specified
    if preset_blend is None:
        preset_blend = preset_glb
    
    # Load preset and clear workbench
    bpyutil.load_blend(preset_blend)
    bpyutil.clear_collection('workbench')
    
    # Set dark background and reduce world lighting to show emission
    if dark_background:
        world = bpy.context.scene.world
        if world is None:
            world = bpy.data.worlds.new("World")
            bpy.context.scene.world = world
        world.use_nodes = True
        bg_node = world.node_tree.nodes.get('Background')
        if bg_node:
            # Dark gray background
            bg_node.inputs['Color'].default_value = (0.05, 0.05, 0.05, 1.0)
            # Reduce world lighting strength so emission is visible
            bg_node.inputs['Strength'].default_value = 0.3
    
    # Load the GLB
    obj = bpyutil.load_glb(glb_path, import_shading=None)
    
    # Create emission texture in Blender
    tex_h, tex_w = emission_texture.shape[:2]
    emission_img = bpy.data.images.new("EmissionTexture", width=tex_w, height=tex_h, alpha=False)
    
    # Convert to Blender format (flatten and add alpha)
    emission_rgba = np.ones((tex_h, tex_w, 4), dtype=np.float32)
    emission_rgba[:, :, :3] = emission_texture.astype(np.float32)
    # Flip vertically for Blender UV coordinates
    emission_rgba = np.flipud(emission_rgba)
    emission_img.pixels = emission_rgba.flatten().tolist()
    
    # Apply emission texture to all materials
    apply_emission_texture_to_materials(obj, emission_img, emission_strength)
    
    # Set camera position and orientation
    bpyutil.set_camera_pos(camera_position)
    # Set camera to look at origin (0,0,0)
    import mathutils
    camera = bpy.context.scene.camera
    # Use look_at matrix to orient camera
    location = mathutils.Vector(camera_position)
    target = mathutils.Vector((0, 0, 0))
    up = mathutils.Vector(camera_up)

    # Create rotation matrix
    direction = (target - location).normalized()
    right = direction.cross(up).normalized()
    up = right.cross(direction).normalized()

    # Build rotation matrix
    rot_matrix = mathutils.Matrix((
        (right.x, up.x, -direction.x, 0),
        (right.y, up.y, -direction.y, 0),
        (right.z, up.z, -direction.z, 0),
        (0, 0, 0, 1)
    ))

    camera.rotation_euler = rot_matrix.to_euler()
    
    # Render
    img = bpyutil.render_scene(
        obj=obj,
        resolution=resolution,
        samples=samples,
        camera_position=camera_position,
        denoise=True,
        shadow_catcher=False
    )
    
    # Composite onto dark background if image has alpha
    if img.shape[-1] == 4:
        # Extract RGB and alpha
        rgb = img[:, :, :3]
        alpha = img[:, :, 3:4]
        
        if dark_background:
            # Dark gray background
            bg_color = np.array([0.05, 0.05, 0.05])
        else:
            # White background
            bg_color = np.array([1.0, 1.0, 1.0])
        
        # Composite: result = rgb * alpha + bg * (1 - alpha)
        bg = np.ones_like(rgb) * bg_color
        img = rgb * alpha + bg * (1 - alpha)
    
    # Save rendered image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    visutil.saveImg(output_path, img)
    
    # Cleanup
    bpyutil.purge_obj(obj)
    bpy.data.images.remove(emission_img)
    
    return img


def get_camera_views():
    """Get camera configurations for multiview rendering."""
    CAMERA_DISTANCE = 2 * math.sqrt(3)  # Same as multiview renderer
    
    views = {
        "front": {"position": (0, -CAMERA_DISTANCE, CAMERA_DISTANCE * 0.3), "up": (0, 0, 1)},
        "back": {"position": (0, CAMERA_DISTANCE, CAMERA_DISTANCE * 0.3), "up": (0, 0, 1)},
        "right": {"position": (CAMERA_DISTANCE, 0, CAMERA_DISTANCE * 0.3), "up": (0, 0, 1)},
        "left": {"position": (-CAMERA_DISTANCE, 0, CAMERA_DISTANCE * 0.3), "up": (0, 0, 1)},
    }
    return views


def render_sample(sample_id, inference_dir, output_dir, glb_path, views, resolution, emission_strength=1.0):
    """Render a single sample with GT and predicted emission from multiple views."""
    sample_infer_dir = inference_dir / sample_id
    sample_output_dir = output_dir / sample_id
    sample_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load emission textures
    gt_emission_path = sample_infer_dir / "gt_emission.png"
    pred_emission_path = sample_infer_dir / "pred_emission.png"
    
    if not gt_emission_path.exists() or not pred_emission_path.exists():
        print(f"  Skipping {sample_id}: Missing emission textures")
        return None
    
    gt_emission = np.array(Image.open(gt_emission_path)).astype(np.float32) / 255.0
    pred_emission = np.array(Image.open(pred_emission_path)).astype(np.float32) / 255.0
    
    print(f"  Rendering {sample_id}...")
    
    results = {'gt': {}, 'pred': {}}
    
    for view_name, view_config in views.items():
        print(f"    View: {view_name}")
        
        # Render with GT emission
        gt_output = sample_output_dir / f"gt_{view_name}.png"
        try:
            render_with_emission(
                str(glb_path),
                gt_emission,
                str(gt_output),
                camera_position=view_config['position'],
                camera_up=view_config['up'],
                resolution=resolution,
                samples=64,
                emission_strength=emission_strength
            )
            results['gt'][view_name] = str(gt_output)
        except Exception as e:
            print(f"      GT render failed: {e}")
        
        # Render with predicted emission
        pred_output = sample_output_dir / f"pred_{view_name}.png"
        try:
            render_with_emission(
                str(glb_path),
                pred_emission,
                str(pred_output),
                camera_position=view_config['position'],
                camera_up=view_config['up'],
                resolution=resolution,
                samples=64,
                emission_strength=emission_strength
            )
            results['pred'][view_name] = str(pred_output)
        except Exception as e:
            print(f"      Pred render failed: {e}")
    
    # Create comparison image (GT row, Pred row)
    create_sample_comparison(sample_output_dir, views)
    
    return results


def create_sample_comparison(sample_dir, views):
    """Create a comparison image for a single sample."""
    gt_images = []
    pred_images = []
    
    for view_name in views.keys():
        gt_path = sample_dir / f"gt_{view_name}.png"
        pred_path = sample_dir / f"pred_{view_name}.png"
        
        if gt_path.exists():
            gt_images.append(np.array(Image.open(gt_path)))
        if pred_path.exists():
            pred_images.append(np.array(Image.open(pred_path)))
    
    if gt_images and pred_images:
        gt_row = np.concatenate(gt_images, axis=1)
        pred_row = np.concatenate(pred_images, axis=1)
        
        # Add spacing between rows
        spacing = np.ones((10, gt_row.shape[1], gt_row.shape[2]), dtype=np.uint8) * 255
        comparison = np.concatenate([gt_row, spacing, pred_row], axis=0)
        
        Image.fromarray(comparison).save(sample_dir / "comparison.png")


def create_overall_grid(output_dir, sample_ids, views):
    """Create an overall comparison grid of all samples."""
    all_rows = []
    
    for sample_id in sample_ids:
        sample_dir = output_dir / sample_id
        comparison_path = sample_dir / "comparison.png"
        
        if comparison_path.exists():
            row_img = np.array(Image.open(comparison_path))
            all_rows.append(row_img)
    
    if all_rows:
        # Add spacing between samples
        spaced_rows = []
        for i, row in enumerate(all_rows):
            spaced_rows.append(row)
            if i < len(all_rows) - 1:
                spacing = np.ones((20, row.shape[1], row.shape[2]), dtype=np.uint8) * 255
                spaced_rows.append(spacing)
        
        grid = np.concatenate(spaced_rows, axis=0)
        
        # Add header
        header_height = 40
        header = np.ones((header_height, grid.shape[1], grid.shape[2]), dtype=np.uint8) * 255
        
        grid_with_header = np.concatenate([header, grid], axis=0)
        
        grid_path = output_dir / "overall_comparison.png"
        Image.fromarray(grid_with_header).save(grid_path)
        print(f"\nSaved overall comparison to: {grid_path}")


def main():
    import sys
    
    # When called from Blender with --, get args after the separator
    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
    else:
        argv = sys.argv[1:]
    
    parser = argparse.ArgumentParser(description='Render inference results with Blender')
    parser.add_argument('--inference-dir', type=str, default='inference_outputs',
                        help='Directory containing inference outputs')
    parser.add_argument('--output-dir', type=str, default='render_outputs',
                        help='Directory to save rendered images')
    parser.add_argument('--parquet-path', type=str,
                        default='/localhome/dya78/code/lightgen/data/baked_uv_local/df_SomgProc_filtered.parquet',
                        help='Path to parquet file with GLB paths')
    parser.add_argument('--glb-root', type=str,
                        default='/cs/3dlg-falas/datasets/TexVerse-1K',
                        help='Root directory of GLB files')
    parser.add_argument('--resolution', type=int, default=512, help='Render resolution')
    parser.add_argument('--emission-strength', type=float, default=1.0,
                        help='Emission strength multiplier (default: 1.0)')
    parser.add_argument('--sample-ids', type=str, nargs='+', default=None,
                        help='Specific sample IDs to render (default: all in inference-dir)')
    args = parser.parse_args(argv)
    
    inference_dir = Path(args.inference_dir)
    output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load parquet to get GLB paths
    df = pd.read_parquet(args.parquet_path)
    
    # Get sample IDs
    if args.sample_ids:
        sample_ids = args.sample_ids
    else:
        # Find all sample directories in inference_dir
        sample_ids = [d.name for d in inference_dir.iterdir() if d.is_dir()]
    
    # Get camera views
    views = get_camera_views()
    
    print("=" * 80)
    print("Rendering Inference Results with Blender")
    print("=" * 80)
    print(f"Inference dir: {inference_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Samples: {len(sample_ids)}")
    print(f"Views: {list(views.keys())}")
    print(f"Resolution: {args.resolution}x{args.resolution}")
    print("=" * 80)
    
    successful_samples = []
    
    # Render each sample
    for sample_id in sample_ids:
        # Get GLB path from parquet
        if sample_id not in df.index:
            print(f"  Skipping {sample_id}: Not found in parquet")
            continue
        
        row = df.loc[sample_id]
        if 'glb_1k_path' not in row or pd.isna(row['glb_1k_path']):
            print(f"  Skipping {sample_id}: No glb_1k_path")
            continue
        
        glb_path = Path(args.glb_root) / row['glb_1k_path']
        if not glb_path.exists():
            print(f"  Skipping {sample_id}: GLB not found at {glb_path}")
            continue
        
        result = render_sample(
            sample_id,
            inference_dir,
            output_dir,
            glb_path,
            views,
            resolution=(args.resolution, args.resolution),
            emission_strength=args.emission_strength
        )

        if result:
            successful_samples.append(sample_id)
    
    # Create overall comparison grid
    if successful_samples:
        create_overall_grid(output_dir, successful_samples, views)
    
    print("\n" + "=" * 80)
    print(f"Rendering complete! {len(successful_samples)}/{len(sample_ids)} samples rendered.")
    print("=" * 80)


if __name__ == "__main__":
    main()
