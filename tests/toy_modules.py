"""A toy backbone and image tokenizer for tests/test_training_hooks.py.

They have the interfaces TEXGenDiffusion calls (the backbone's forward signature and its
(output, addition_info) return, the tokenizer's process_text/process_image) and nothing else:
no torchsparse, no CUDA and no CLIP download, so the training hooks can run on the CPU.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn

from spuv.utils.base import BaseModule


class ToyNet(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        in_channels: int = 13
        out_channels: int = 3

    cfg: Config

    def configure(self):
        super().configure()
        self.conv = nn.Conv2d(self.cfg.in_channels, 8, 3, padding=1)
        self.emb = nn.Linear(768 * 2 + 1, 8)
        self.out = nn.Conv2d(8, self.cfg.out_channels, 3, padding=1)

    def forward(self, x_dense, mask_map, position_map, timestep, clip_embeddings, mesh, image_info,
                data_normalization, condition_drop):
        bt = image_info["baked_texture"]
        if data_normalization and bt.max() <= 1.0 and bt.min() >= 0.0:
            bt = bt * 2.0 - 1.0
        x = torch.cat([x_dense, position_map, bt, image_info["baked_weights"]], dim=1)
        assert x.shape[1] == self.cfg.in_channels, x.shape
        cd = condition_drop.unsqueeze(-1)
        embs = [(1 - cd) * e for e in clip_embeddings]
        e = self.emb(torch.cat(embs + [timestep[:, None].to(x.dtype)], dim=1))
        h = torch.relu(self.conv(x) + e[:, :, None, None])
        return self.out(h), {"baked_texture": bt}


class ToyTok(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = ""

    cfg: Config

    def process_text(self, prompts):
        return torch.full((len(prompts), 768), 0.1)

    def process_image(self, img):
        return img.float().flatten(1).mean(1, keepdim=True).expand(-1, 768).contiguous()
