#!/usr/bin/env python3
"""
Render inference results with GT and predicted emission textures.
Uses xgutils/bpyutil for Blender rendering.

Usage:
    python render_inference_multiview.py --inference-dir inference_outputs --output-dir render_outputs
"""
import os
import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

# Add xgutils to path
sys.path.insert(0, str(Path(__file__).parent.parent / "external"))

def render_with_emission_texture(
    glb_path: str,
    emission_texture: np.ndarray,
    output_path: str,
    camera_position: tuple = (2, 2, 2),
    camera_up: tuple = (0, 0, 1),
    resolution: tuple = (512, 512),
    samples: int = 64,
    preset_blend: str = None
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
    """
    from xgutils import bpyutil
    from xgutils.vis import visutil
    import bpy
    
    # Get default preset if not specified
    if preset_blend is None:
        from xgutils.miscutil import get_asset_path
        preset_blend = get_asset_path('preset_glb.blend')
    
    # Load preset and clear workbench
    bpyutil.load_blend(preset_blend)
    bpyutil.clear_collection('workbench')
    
    # Load the GLB
    obj = bpyutil.load_glb(glb_path, import_shading=None)
    
    # Create emission texture in Blender
    tex_h, tex_w = emission_texture.shape[:2]
    emission_img = bpy.data.images.new("EmissionTexture", width=tex_w, height=tex_h, alpha=False)
    
    # Convert to Blender format (flatten and add alpha)
    emission_rgba = np.ones((tex_h, tex_w, 4), dtype=np.float32)
    emission_rgba[:, :, :3] = emission_texture
    # Flip vertically for Blender UV coordinates
    emission_rgba = np.flipud(emission_rgba)
    emission_img.pixels = emission_rgba.flatten()
    
    # Apply emission texture to all materials
    for mat in obj.data.materials:
        if mat is None:
            continue
        
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        
        # Find or create Principled BSDF
        bsdf_node = None
        for node in nodes:
            if node.type == 'BSDF_PRINCIPLED':
                bsdf_node = node
                break
        
        if bsdf_node is None:
            continue
        
        # Create texture node for emission
        tex_node = nodes.new(type='ShaderNodeTexImage')
        tex_node.image = emission_img
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
        bsdf_node.inputs['Emission Strength'].default_value = 1.0
    
    # Render
    img = bpyutil.render_scene(
        obj=obj,
        resolution=resolution,
        samples=samples,
        camera_position=camera_position,
        denoise=True,
        shadow_catcher=False
    )
    
    # Save rendered image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    visutil.saveImg(output_path, img)
    
    # Cleanup
    bpyutil.purge_obj(obj)
    bpy.data.images.remove(emission_img)
    
    return img


def render_inference_sample(
    sample_id: str,
    inference_dir: Path,
    data_root: Path,
    output_dir: Path,
    views: dict,
    resolution: tuple = (512, 512)
):
    """Render a single inference sample from multiple views."""
    import pandas as pd
    
    sample_dir = inference_dir / sample_id
    
    # Load textures
    gt_emission_path = sample_dir / "gt_emission.png"
    pred_emission_path = sample_dir / "pred_emission.png"
    
    if not gt_emission_path.exists() or not pred_emission_path.exists():
        print(f"  Skipping {sample_id}: Missing emission textures")
        return None
    
    gt_emission = np.array(Image.open(gt_emission_path)).astype(np.float32) / 255.0
    pred_emission = np.array(Image.open(pred_emission_path)).astype(np.float32) / 255.0
    
    # Find the GLB path from parquet
    parquet_path = data_root / "df_SomgProc_final.parquet"
    if not parquet_path.exists():
        parquet_path = data_root / "df_SomgProc_filtered.parquet"
    
    df = pd.read_parquet(parquet_path)
    if sample_id not in df.index:
        print(f"  Skipping {sample_id}: Not found in parquet")
        return None
    
    row = df.loc[sample_id]
    
    # Construct GLB path
    if 'glb_1k_path' in row:
        glb_path = Path("/3dlg-falas/project/omages/datasets/TexVerse/TexVerse-1K") / row['glb_1k_path']
    else:
        print(f"  Skipping {sample_id}: No glb_1k_path")
        return None
    
    if not glb_path.exists():
        print(f"  Skipping {sample_id}: GLB not found at {glb_path}")
        return None
    
    print(f"  Rendering {sample_id}...")
    
    results = {'sample_id': sample_id, 'views': {}}
    
    for view_name, view_config in views.items():
        print(f"    View: {view_name}")
        
        # Render with GT emission
        gt_output = output_dir / sample_id / f"gt_emission_{view_name}.png"
        try:
            render_with_emission_texture(
                str(glb_path),
                gt_emission,
                str(gt_output),
                camera_position=view_config['position'],
                camera_up=view_config['up'],
                resolution=resolution
            )
        except Exception as e:
            print(f"      GT render failed: {e}")
        
        # Render with predicted emission
        pred_output = output_dir / sample_id / f"pred_emission_{view_name}.png"
        try:
            render_with_emission_texture(
                str(glb_path),
                pred_emission,
                str(pred_output),
                camera_position=view_config['position'],
                camera_up=view_config['up'],
                resolution=resolution
            )
        except Exception as e:
            print(f"      Pred render failed: {e}")
        
        results['views'][view_name] = {
            'gt': str(gt_output),
            'pred': str(pred_output)
        }
    
    return results


def create_comparison_grid(output_dir: Path, sample_ids: list, views: dict):
    """Create a comparison grid of all rendered samples."""
    import cv2
    
    # Collect all rendered images
    all_rows = []
    
    for sample_id in sample_ids:
        sample_dir = output_dir / sample_id
        
        row_gt = []
        row_pred = []
        
        for view_name in views.keys():
            gt_path = sample_dir / f"gt_emission_{view_name}.png"
            pred_path = sample_dir / f"pred_emission_{view_name}.png"
            
            if gt_path.exists():
                gt_img = np.array(Image.open(gt_path))
                row_gt.append(gt_img)
            
            if pred_path.exists():
                pred_img = np.array(Image.open(pred_path))
                row_pred.append(pred_img)
        
        if row_gt and row_pred:
            # Stack views horizontally for GT and Pred
            gt_row = np.concatenate(row_gt, axis=1)
            pred_row = np.concatenate(row_pred, axis=1)
            # Stack GT and Pred vertically for this sample
            sample_block = np.concatenate([gt_row, pred_row], axis=0)
            all_rows.append(sample_block)
    
    if all_rows:
        # Add spacing between samples
        spacing = 10
        spaced_rows = []
        for i, row in enumerate(all_rows):
            spaced_rows.append(row)
            if i < len(all_rows) - 1:
                spacer = np.ones((spacing, row.shape[1], row.shape[2]), dtype=np.uint8) * 255
                spaced_rows.append(spacer)
        
        grid = np.concatenate(spaced_rows, axis=0)
        
        # Save grid
        grid_path = output_dir / "comparison_grid.png"
        Image.fromarray(grid).save(grid_path)
        print(f"\nSaved comparison grid to: {grid_path}")


def main():
    parser = argparse.ArgumentParser(description='Render inference results with emission textures')
    parser.add_argument('--inference-dir', type=str, default='inference_outputs',
                        help='Directory containing inference outputs')
    parser.add_argument('--output-dir', type=str, default='render_outputs',
                        help='Directory to save rendered images')
    parser.add_argument('--data-root', type=str, 
                        default='/localhome/dya78/code/lightgen/data/baked_uv',
                        help='Root directory of baked UV data')
    parser.add_argument('--resolution', type=int, default=512, help='Render resolution')
    parser.add_argument('--sample-ids', type=str, nargs='+', default=None,
                        help='Specific sample IDs to render (default: all in inference-dir)')
    args = parser.parse_args()
    
    inference_dir = Path(args.inference_dir)
    output_dir = Path(args.output_dir)
    data_root = Path(args.data_root)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define camera views
    import math
    CAMERA_DISTANCE = 2 * math.sqrt(3)
    
    views = {
        "front": {"position": (0, -CAMERA_DISTANCE, 0), "up": (0, 0, 1)},
        "back": {"position": (0, CAMERA_DISTANCE, 0), "up": (0, 0, 1)},
        "right": {"position": (CAMERA_DISTANCE, 0, 0), "up": (0, 0, 1)},
        "top": {"position": (0, 0, CAMERA_DISTANCE), "up": (0, 1, 0)},
    }
    
    # Get sample IDs
    if args.sample_ids:
        sample_ids = args.sample_ids
    else:
        # Find all sample directories in inference_dir
        sample_ids = [d.name for d in inference_dir.iterdir() if d.is_dir()]
    
    print("=" * 80)
    print("Rendering Inference Results with Emission Textures")
    print("=" * 80)
    print(f"Inference dir: {inference_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Samples: {len(sample_ids)}")
    print(f"Views: {list(views.keys())}")
    print("=" * 80)
    
    # Render each sample
    for sample_id in sample_ids:
        render_inference_sample(
            sample_id,
            inference_dir,
            data_root,
            output_dir,
            views,
            resolution=(args.resolution, args.resolution)
        )
    
    # Create comparison grid
    create_comparison_grid(output_dir, sample_ids, views)
    
    print("\n" + "=" * 80)
    print("Rendering complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
