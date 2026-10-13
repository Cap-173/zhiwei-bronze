"""PointNet++-inspired single-scale grouping; pure Torch CPU implementation.

Reference: charlesq34/pointnet2 utils/pointnet_util.py sample_and_group,
pointnet_sa_module and sample_and_group_all. This is a compact adaptation.
"""
import torch
from torch import nn


@torch.no_grad()
def local_groups(points, anchors, neighbors):
    from completion import farthest_subset
    if points.ndim != 3 or points.shape[-1] != 3 or min(anchors, neighbors) < 1 or max(anchors, neighbors) > points.shape[1]:
        raise ValueError('邻域点数或形状无效。')
    # Canonical tie breaking, then nearest-to-hole start: permutation independent
    # for distinct coordinates. Coincident points are interchangeable.
    ordered=points
    for axis in (2,1,0):
        indices=ordered[:,:,axis].argsort(dim=1,stable=True)
        ordered=ordered.gather(1,indices[:,:,None].expand(-1,-1,3))
    indices=ordered.square().sum(-1).argsort(dim=1,stable=True)
    ordered=ordered.gather(1,indices[:,:,None].expand(-1,-1,3))
    centers=farthest_subset(ordered,anchors)
    distances=torch.cdist(centers,ordered,compute_mode='donot_use_mm_for_euclid_dist')
    indices=distances.argsort(dim=-1,stable=True)[:,:,:neighbors]
    batch=torch.arange(points.shape[0],device=points.device)[:,None,None]
    grouped=ordered[batch,indices]
    return centers,grouped-centers[:,:,None,:]


class NeighborhoodEncoder(nn.Module):
    def __init__(self, mode='local', anchors=32, neighbors=16):
        super().__init__()
        if mode not in ('local','pointwise_xyz') or min(anchors,neighbors)<1:
            raise ValueError('未知邻域编码配置。')
        self.mode,self.anchors,self.neighbors=mode,anchors,neighbors
        self.point_mlp=nn.Sequential(nn.Linear(3,64),nn.ReLU(),nn.Linear(64,128),nn.ReLU())
        self.region_mlp=nn.Sequential(nn.Linear(131,128),nn.ReLU())

    def forward(self, relative):
        if self.mode=='local':
            centers,offsets=local_groups(relative,self.anchors,self.neighbors)
            # Input coordinates are data; grouping is a discrete nontrainable step.
            features=self.point_mlp(offsets).amax(dim=2)
        else:
            centers=relative
            features=self.point_mlp(relative)
        return self.region_mlp(torch.cat((centers,features),dim=-1)).amax(dim=1)
