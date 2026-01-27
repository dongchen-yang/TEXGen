import json
import math
import os
import random
from dataclasses import dataclass, field
import math

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchvision.transforms import v2 as transform
from PIL import Image
import imageio
import cv2
from torch.utils.data import DataLoader, Dataset

import spuv
from spuv.utils.config import parse_structured
from spuv.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
)
from spuv.utils.typing import *
from spuv.utils.mesh_utils import *
from spuv.data.camera_strategy import camera_functions


def _parse_scene_list(scene_path):
    """Parse JSONL file containing scene information"""
    data = []
    with open(scene_path, 'r') as file:
        for line in file:
            try:
                json_data = json.loads(line.strip())
                data.append(json_data)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e} in line: {line}")
    return data


def decode_uint16_to_float(data, min_val=-2.0, max_val=2.0):
    """Decode uint16 data to float in range [min_val, max_val]"""
    return (data.astype(np.float32) / 65535.0) * (max_val - min_val) + min_val


def decode_uint8_to_float(data):
    """Decode uint8 data to float in range [0, 1]"""
    return data.astype(np.float32) / 255.0


@dataclass
class LightGenDataModuleConfig:
    data_root: str = "/localhome/dya78/code/lightgen/data/baked_uv_local"
    parquet_file: str = "/localhome/dya78/code/lightgen/data/baked_uv_local/df_SomgProc_filtered.parquet"
    scene_list: str = ""
    eval_scene_list: str = ""
    repeat: int = 1  # for debugging purpose
    camera_strategy: str = "strategy_1"
    eval_camera_strategy: str = "strategy_1"
    height: int = 128
    width: int = 128
    cond_views: int = 1
    sup_views: int = 4
    uv_height: int = 512
    uv_width: int = 512
    
    # Can be either: tuple (start, end) or list of indices or path to JSON split file
    train_indices: Any = None
    val_indices: Any = None
    test_indices: Any = None

    batch_size: int = 1
    num_workers: int = 4

    eval_height: int = 512
    eval_width: int = 512
    eval_batch_size: int = 1
    eval_cond_views: int = 1
    eval_sup_views: int = 4

    vertex_transformation: bool = False


class LightGenDataset(Dataset):
    def __init__(self, cfg: Any, split: str = "train") -> None:
        super().__init__()
        assert split in ["train", "val", "test"]
        self.cfg: LightGenDataModuleConfig = cfg
        self.split = split

        # Load sample list from parquet file (much faster than scanning!)
        self.all_samples = self._load_from_parquet()
        
        # Apply split indices
        if self.split == "train" and self.cfg.train_indices is not None:
            self.all_samples = self._apply_indices(self.all_samples, self.cfg.train_indices)
            self.all_samples = self.all_samples * self.cfg.repeat
        elif self.split == "val" and self.cfg.val_indices is not None:
            self.all_samples = self._apply_indices(self.all_samples, self.cfg.val_indices)
        elif self.split == "test" and self.cfg.test_indices is not None:
            self.all_samples = self._apply_indices(self.all_samples, self.cfg.test_indices)
    
    def _apply_indices(self, samples, indices):
        """Apply indices to samples - supports tuple, list, or JSON file path"""
        if isinstance(indices, tuple):
            # Range-based indexing: (start, end)
            return samples[indices[0]:indices[1]]
        elif isinstance(indices, list):
            # List of specific indices
            return [samples[i] for i in indices]
        elif isinstance(indices, str):
            # Path to JSON file with splits
            import json
            with open(indices, 'r') as f:
                split_data = json.load(f)
            # Extract indices for this split
            split_indices = split_data.get(self.split, {}).get('indices', [])
            return [samples[i] for i in split_indices]
        else:
            return samples

    def _load_from_parquet(self):
        """Load sample list from parquet file (much faster than scanning)"""
        import pandas as pd
        
        # Read parquet file
        df = pd.read_parquet(self.cfg.parquet_file)
        
        # Filter for successful samples only
        if 'success' in df.columns:
            df = df[df['success'] == True]
        
        samples = []
        data_root = self.cfg.data_root
        
        # Convert to sample list
        for sample_id, row in df.iterrows():
            # Get relative path from ditem_dir column
            ditem_dir = row.get('ditem_dir', None)
            if ditem_dir is None:
                # Fallback: construct from sample_id
                ditem_dir = f"{sample_id[:3]}-{sample_id[3:6]}/{sample_id}"
            
            sample_path = os.path.join(data_root, ditem_dir)
            npz_file = os.path.join(sample_path, "somage.npz")
            
            samples.append({
                "sample_id": sample_id,
                "path": sample_path,
                "npz_file": npz_file,
            })
        
        print(f"Loaded {len(samples)} samples from parquet file for {self.split} split")
        return samples

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, index):
        try:
            return self.try_get_item(index)
        except Exception as e:
            print(f"Failed to load {index}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def vertex_transform(self, mesh):
        """Transform mesh vertices to canonical space"""
        pre_vertices = mesh['v_pos']

        vertices = torch.clone(pre_vertices)
        vertices[:, 1] = -pre_vertices[:, 2]  # -z --> y
        vertices[:, 2] = pre_vertices[:, 1]  # y --> z

        bounding_box_max = vertices.max(0)[0]
        bounding_box_min = vertices.min(0)[0]
        mesh_scale = 1.0
        scale = mesh_scale / ((bounding_box_max - bounding_box_min).max() + 1e-6)
        center_offset = (bounding_box_max + bounding_box_min) * 0.5
        vertices = (vertices - center_offset) * scale

        mesh['v_pos'] = vertices
        return mesh

    def try_get_item(self, index):
        sample_info = self.all_samples[index]
        sample_id = sample_info["sample_id"]
        npz_file = sample_info["npz_file"]
        
        # Load the NPZ file - extract data immediately to allow file to be closed
        npz_data = np.load(npz_file)
        occupancy_np = npz_data['occupancy'].copy()
        position_np = npz_data['position'].copy()
        objnormal_np = npz_data['objnormal'].copy()
        color_np = npz_data['color'].copy()
        metal_np = npz_data['metal'].copy()
        rough_np = npz_data['rough'].copy()
        emission_color_np = npz_data['emission_color'].copy()
        npz_data.close()  # Explicitly close to free file handles
        
        # Load pre-rendered thumbnail for CLIP conditioning (from local data directory)
        # Thumbnails are stored in data/baked_uv_local/thumbnails/
        thumbnail_path = os.path.join(self.cfg.data_root, "thumbnails", f"{sample_id}.png")
        thumbnail = None
        if os.path.exists(thumbnail_path):
            from PIL import Image
            import torchvision.transforms.functional as TF
            with Image.open(thumbnail_path) as thumbnail_pil:
                thumbnail_img = thumbnail_pil.convert('RGB')
                # Resize to fixed size (224x224) for batching - CLIP will resize anyway
                thumbnail_img = TF.resize(thumbnail_img, [224, 224], interpolation=TF.InterpolationMode.BILINEAR)
                thumbnail = torch.from_numpy(np.array(thumbnail_img)).float() / 255.0  # [224, 224, 3], normalize to [0, 1]
                thumbnail = thumbnail.unsqueeze(0)  # [1, 224, 224, 3]
        
        # Extract relevant data and convert to torch tensors
        occupancy = torch.from_numpy(occupancy_np).float()  # [512, 512, 1]
        position = torch.from_numpy(decode_uint16_to_float(position_np)).float()  # [512, 512, 3]
        objnormal = torch.from_numpy(decode_uint16_to_float(objnormal_np, -1.0, 1.0)).float()  # [512, 512, 3]
        color = torch.from_numpy(decode_uint8_to_float(color_np)).float()  # [512, 512, 3] (albedo)
        metal = torch.from_numpy(decode_uint8_to_float(metal_np)).float()  # [512, 512, 1]
        rough = torch.from_numpy(decode_uint8_to_float(rough_np)).float()  # [512, 512, 1]
        emission_color = torch.from_numpy(decode_uint8_to_float(emission_color_np)).float()  # [512, 512, 3]
        
        # Clean up numpy arrays to free memory
        del occupancy_np, position_np, objnormal_np, color_np, metal_np, rough_np, emission_color_np
        
        # Rearrange from [H, W, C] to [C, H, W]
        occupancy = occupancy.permute(2, 0, 1)  # [1, 512, 512]
        position = position.permute(2, 0, 1)  # [3, 512, 512]
        objnormal = objnormal.permute(2, 0, 1)  # [3, 512, 512]
        color = color.permute(2, 0, 1)  # [3, 512, 512]
        metal = metal.permute(2, 0, 1)  # [1, 512, 512]
        rough = rough.permute(2, 0, 1)  # [1, 512, 512]
        emission_color = emission_color.permute(2, 0, 1)  # [3, 512, 512]
        
        # Normalize emission_color to [-1, 1] for diffusion
        emission_color = emission_color * 2.0 - 1.0
        
        # Create input tensor: [position(3), normal(3), color(3), metal(1), rough(1)] = 11 channels
        input_tensor = torch.cat([position, objnormal, color, metal, rough], dim=0)
        
        # Resize UV data to match configured resolution if needed
        current_h, current_w = 512, 512
        target_h, target_w = self.cfg.uv_height, self.cfg.uv_width
        
        if current_h != target_h or current_w != target_w:
            import torch.nn.functional as F
            # Resize all UV maps to target resolution
            # Use bilinear for continuous values, nearest for masks
            occupancy = F.interpolate(occupancy.unsqueeze(0), size=(target_h, target_w), mode='nearest').squeeze(0)
            position = F.interpolate(position.unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze(0)
            objnormal = F.interpolate(objnormal.unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze(0)
            color = F.interpolate(color.unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze(0)
            metal = F.interpolate(metal.unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze(0)
            rough = F.interpolate(rough.unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze(0)
            emission_color = F.interpolate(emission_color.unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze(0)
            input_tensor = F.interpolate(input_tensor.unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze(0)
        
        # Construct a mesh with proper UV coordinates for feature baking
        H, W = target_h, target_w
        
        # Create a grid of UV coordinates
        u = torch.linspace(0, 1, W)
        v = torch.linspace(0, 1, H)
        v_grid, u_grid = torch.meshgrid(v, u, indexing='ij')
        
        # Flatten UV coordinates
        uv_coords = torch.stack([u_grid.flatten(), v_grid.flatten()], dim=1)  # [H*W, 2]
        
        # Create triangles for the UV grid (two triangles per quad)
        triangles = []
        triangle_uvs = []
        for i in range(H-1):
            for j in range(W-1):
                # First triangle
                v0 = i * W + j
                v1 = i * W + (j + 1)
                v2 = (i + 1) * W + j
                triangles.append([v0, v1, v2])
                triangle_uvs.append([v0, v1, v2])
                
                # Second triangle
                v0 = i * W + (j + 1)
                v1 = (i + 1) * W + (j + 1)
                v2 = (i + 1) * W + j
                triangles.append([v0, v1, v2])
                triangle_uvs.append([v0, v1, v2])
        
        triangles = torch.tensor(triangles, dtype=torch.long)
        triangle_uvs = torch.tensor(triangle_uvs, dtype=torch.long)
        
        mesh = {
            'v_pos': position.permute(1, 2, 0).reshape(-1, 3),  # [H*W, 3]
            't_pos_idx': triangles,  # [N_triangles, 3]
            'v_tex': uv_coords,  # [H*W, 2]
            '_v_tex': uv_coords,  # Also add _v_tex key (used by feature baking)
            't_tex_idx': triangle_uvs,  # [N_triangles, 3]
            '_t_tex_idx': triangle_uvs,  # Also add _t_tex_idx key (used by rendering)
        }
        
        # Camera setup for conditioning (if needed)
        if self.split == "train":
            height = self.cfg.height
            width = self.cfg.width
            cond_views = self.cfg.cond_views
            sup_views = self.cfg.sup_views
            camera_strategy = self.cfg.camera_strategy
        else:
            height = self.cfg.eval_height
            width = self.cfg.eval_width
            cond_views = self.cfg.eval_cond_views
            sup_views = self.cfg.eval_sup_views
            camera_strategy = self.cfg.eval_camera_strategy

        # Generate camera matrices (dummy for now since we're working in UV space)
        camera_info = self.get_camera_info(cond_views, sup_views, camera_strategy, height, width)
        
        return_dict = {
            "scene_id": sample_id,
            "index": index,
            "height": height,
            "width": width,
            "uv_height": self.cfg.uv_height,
            "uv_width": self.cfg.uv_width,
            
            # UV space data (resized to match configured resolution)
            "input_tensor": input_tensor,  # [11, H, W] where H=uv_height, W=uv_width
            "position_map": position,  # [3, H, W]
            "normal_map": objnormal,  # [3, H, W]
            "albedo_map": color,  # [3, H, W]
            "metal_map": metal,  # [1, H, W]
            "rough_map": rough,  # [1, H, W]
            "mask_map": occupancy,  # [1, H, W]
            "gt_emission": emission_color,  # [3, H, W], normalized to [-1, 1]
            
            # Pre-rendered thumbnail for CLIP
            "thumbnail": thumbnail,  # [1, 224, 224, 3] or None
            
            # Mesh and camera info
            "mesh": mesh,
            **camera_info,
        }
        
        return return_dict

    def get_camera_info(self, cond_views, sup_views, camera_strategy, height, width):
        """Generate dummy camera matrices (not used for UV-space training but needed for compatibility)"""
        total_views = cond_views + sup_views
        
        # Create dummy camera matrices - just identity transforms
        # These aren't used in UV-space training but are expected by the system
        c2ws = torch.eye(4).unsqueeze(0).repeat(total_views, 1, 1).float()
        
        # Create projection matrices
        fov = 49.1  # degrees
        projection_matrix = get_projection_matrix(fov, width / height, 0.1, 100.0)
        projection_matrix = projection_matrix.unsqueeze(0).repeat(total_views, 1, 1)
        
        # Create MVP matrices
        mvp_mtx = get_mvp_matrix(c2ws, projection_matrix)
        
        # Split into condition and supervision views
        mvp_mtx_cond = mvp_mtx[:cond_views]
        mvp_mtx_sup = mvp_mtx[cond_views:]
        c2w_cond = c2ws[:cond_views]
        c2w_sup = c2ws[cond_views:]
        
        return {
            "mvp_mtx": mvp_mtx,
            "mvp_mtx_cond": mvp_mtx_cond,
            "mvp_mtx_sup": mvp_mtx_sup,
            "c2w": c2ws,
            "c2w_cond": c2w_cond,
            "c2w_sup": c2w_sup,
        }


class LightGenDataModule(pl.LightningDataModule):
    cfg: LightGenDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(LightGenDataModuleConfig, cfg)

    def setup(self, stage=None):
        if stage in [None, "fit"]:
            self.train_dataset = LightGenDataset(self.cfg, split="train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = LightGenDataset(self.cfg, split="val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = LightGenDataset(self.cfg, split="test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, shuffle=False):
        return DataLoader(
            dataset,
            num_workers=self.cfg.num_workers,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
        )

    def train_dataloader(self):
        return self.general_loader(
            self.train_dataset, batch_size=self.cfg.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return self.general_loader(
            self.val_dataset, batch_size=self.cfg.eval_batch_size
        )

    def test_dataloader(self):
        return self.general_loader(
            self.test_dataset, batch_size=self.cfg.eval_batch_size
        )

    def predict_dataloader(self):
        return self.general_loader(
            self.test_dataset, batch_size=self.cfg.eval_batch_size
        )

    def collate_fn(self, batch):
        """Custom collate function to handle None values"""
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None
        
        # Stack all tensors
        collated = {}
        for key in batch[0].keys():
            if isinstance(batch[0][key], torch.Tensor):
                collated[key] = torch.stack([item[key] for item in batch], dim=0)
            elif isinstance(batch[0][key], dict):
                # For mesh dict, keep as list
                collated[key] = [item[key] for item in batch]
            else:
                collated[key] = [item[key] for item in batch]
        
        return collated

