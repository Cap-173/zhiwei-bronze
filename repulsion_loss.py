"""PU-Net inspired repulsion; hole-local units, exact four nearest nonself neighbors.

Author code: yulequan/PU-Net code/model_utils.py get_repulsion_loss4.
We replace capped ball-query/padding with exact kNN, retaining its energy.
"""
import torch


def repulsion_per_sample(prediction, region, bandwidth=0.03, neighbors=4):
    local = (prediction.float() - region[:, None, :3]) / region[:, None, 3:4]
    squared = torch.cdist(local, local, compute_mode='donot_use_mm_for_euclid_dist').square()
    mask = torch.eye(local.shape[1], device=local.device, dtype=torch.bool)[None]
    squared = squared.masked_fill(mask, float('inf')).topk(neighbors, largest=False).values
    distance = squared.clamp_min(1e-12).sqrt()
    return (0.07 - distance * torch.exp(-squared / bandwidth**2)).mean(dim=(1, 2))
