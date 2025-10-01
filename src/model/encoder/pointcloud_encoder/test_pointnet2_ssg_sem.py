import torch

from pointnet2_test.models.pointnet2_ssg_sem_plus import PointNet2SemSegSSGPlus

b = 3

n = 10000

feat_dim = 3

xyz = torch.rand(b, n, 3)
feats = torch.rand(b, n, feat_dim)

pointcloud = torch.cat((xyz, feats), dim=-1).cuda()
pointnet_seg = PointNet2SemSegSSGPlus().cuda()

out = pointnet_seg(pointcloud)
