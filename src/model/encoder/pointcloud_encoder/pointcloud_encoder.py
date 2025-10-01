from dataclasses import dataclass, field
from typing import List


import torch.nn as nn

import sys
sys.path.append('/ssdwork/liuxiang/noposplat_private/src/model/encoder/pointcloud_encoder')
from .layer import *

@dataclass
class PointCloudEncoderCfg:
    n_levels: int
    
    aggregate: str
    norm: str
    actv: str
    res: bool
    
    in_conv: str
    in_dim: int
    in_radius: float
    in_k: int
    
    block: str
    scale_factor: List[int]
    dims: List[int]
    depth: List[int]
    radius: List[List[List[float]]]
    k: List[List[List[int]]]
    
    out_relu: bool
    up: str
    

class PointCloudEncoder(nn.Module):

    def __init__(self, cfg: PointCloudEncoderCfg, in_dim: int):
        super(PointCloudEncoder, self).__init__()

        self.n_levels = cfg.n_levels
        assert len(cfg.scale_factor) == self.n_levels
        assert len(cfg.depth) == self.n_levels
        assert len(cfg.dims) == self.n_levels
        assert len(cfg.k) == self.n_levels
        assert len(cfg.radius) == self.n_levels

        self.in_conv = GCNLayer(
            cfg.in_conv,
            in_dim=in_dim, 
            out_dim=cfg.in_dim, 
            scale_factor=1,
            radius=cfg.in_radius, 
            k=cfg.in_k, 
            aggregate=cfg.aggregate,
            norm=cfg.norm, 
            actv=cfg.actv,
            res=False
        )

        self.blocks = nn.ModuleList()
        in_dim = cfg.in_dim

        for i in range(self.n_levels):
            blocks = nn.ModuleList()
            scale_factor = cfg.scale_factor[i]
            out_dim = cfg.dims[i]
            for j in range(cfg.depth[i]):
                blocks.append(
                    GCNBlock(
                        cfg.block,
                        in_dim=in_dim, 
                        out_dim=out_dim,
                        scale_factor=scale_factor,
                        radius=cfg.radius[i][j],
                        k=cfg.k[i][j],
                        aggregate=cfg.aggregate,
                        norm=cfg.norm, 
                        actv=cfg.actv,
                        res=cfg.res
                    )
                )
                scale_factor = 1
                in_dim = out_dim
            self.blocks.append(blocks)

        self.out_fc = nn.Conv1d(in_dim, in_dim, 1)
        self.out_relu = cfg.out_relu

    def _build_pyramid(self, xyz, feats):
        xyz_list, feats_list = [xyz], [feats]
        for i in range(self.n_levels):
            blocks = self.blocks[i]
            for j in range(len(blocks)):
                xyz, feats = blocks[j](xyz, feats)
            xyz_list.append(xyz)
            feats_list.append(feats)
        feats_list[-1] = self.out_fc(feats_list[-1])
        if self.out_relu:
            feats_list[-1] = F.relu(feats_list[-1], inplace=True)
        return xyz_list, feats_list

    def forward(self, xyz, rgb, up=True):
        """
        Args:
            xyz (float tensor, (bs, p, 3)): point coordinates.
            rgb (float tensor, (bs, 3, p)): point RGB values.
            up (bool): if True, upsample output features to input resolution.
        
        Returns:
            output_dict (dict):
                xyz_list (float tensor list): point NDC coordinates at all levels.
            feats (float tensor, (bs, c, p)): output features.
        """
        assert xyz.size(1) == rgb.size(2), \
            ('[ERROR] point cloud size ({:d}) and number of RGB values ({:d}) '
             'must match'.format(xyz.size(1), rgb.size(2))
            )
        output_dict = dict()

        _, feats = self.in_conv(xyz, rgb)
        xyz_list, feats_list = self._build_pyramid(xyz, feats)
        output_dict['xyz_list'] = xyz_list
        output_dict['feats_list'] = feats_list

        return output_dict

if __name__ == "__main__":
        
    pcd_config = {
        "n_levels": 2,
        "aggregate": "max",
        "norm": "batch",
        "actv": "relu",
        "res": True,
        "in_conv": "mr",
        "in_dim": 64,
        "in_radius": 0.015,
        "in_k": 16,
        "block": "mr",
        "scale_factor": [4, 4],
        "dims": [128, 256],
        "depth": [1, 1],
        "radius": [[[0.015, 0.025]], [[0.025, 0.05]]],
        "k": [[[16, 16]], [[16, 16]]],
        "out_relu": False,
        "up": "linear"
    }
    
    encoder = PointCloudEncoder(pcd_config).cuda()
    
    b = 2
    p = 10000
    
    xyz = torch.rand(b, p, 3).cuda()
    rgb = torch.rand(b, 3, p).cuda()
    up = True
    
    out = encoder(xyz, rgb, up)