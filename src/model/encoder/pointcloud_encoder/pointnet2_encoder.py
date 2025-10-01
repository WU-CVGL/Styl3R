from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn

import sys
sys.path.append('/ssdwork/liuxiang/noposplat_private/src/model/encoder/pointcloud_encoder')
# from pointnet2_ops_lib.pointnet2_modules import PointnetSAModule

@dataclass
class Pointnet2EncoderCfg:
    pass
    

class Pointnet2Encoder(nn.Module):

    # def __init__(self, cfg: Pointnet2EncoderCfg, in_dim: int):
    def __init__(self, in_dim: int):
        super(Pointnet2Encoder, self).__init__()
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=65526,
                radius=0.02,
                nsample=32,
                mlp=[in_dim, 64],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=32768,
                radius=0.04,
                nsample=32,
                mlp=[64, 128],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=16384,
                radius=0.08,
                nsample=64,
                mlp=[128, 256],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=8192,
                radius=0.12,
                nsample=64,
                mlp=[256, 256],
                use_xyz=True,
            )
        )
    
    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features
    
    def forward(self, xyz, features):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """

        assert xyz.size(1) == features.size(2), \
            ('[ERROR] point cloud size ({:d}) and number of RGB values ({:d}) '
             'must match'.format(xyz.size(1), features.size(2))
            )
        
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        return {'xyz_list': l_xyz, 'feats_list': l_features}

if __name__ == "__main__":
        
    pointnet2_config = {}
    
    b = 2
    n = 131_072
    feat_dim = 256
        
    xyz = torch.rand(b, n, 3).cuda()
    feats = torch.rand(b, n, feat_dim).cuda()
    
    pointcloud = torch.cat((xyz, feats), dim=-1).cuda()
    
    encoder = Pointnet2Encoder(pointnet2_config, feat_dim).cuda()
    
    out = encoder(pointcloud)