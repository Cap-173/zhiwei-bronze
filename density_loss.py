"""等点数DCD，依据Wu等NeurIPS 2021公式及作者calc_dcd实现。

Source: https://github.com/wutong16/Density_aware_Chamfer_Distance/blob/main/utils_v2/model_utils.py
Pure PyTorch adaptation; squared distances, detached nearest-neighbor counts.
"""
import math
import torch


def density_chamfer(prediction, target, alpha=50., exponent=0.5):
    if prediction.ndim != 3 or prediction.shape != target.shape or prediction.shape[-1] != 3 or not prediction.shape[1]:
        raise ValueError('DCD要求非空、等点数的B×N×3预测与目标。')
    if not math.isfinite(alpha) or alpha <= 0 or not math.isfinite(exponent) or not 0 <= exponent <= 1:
        raise ValueError('DCD参数无效。')
    distance = torch.cdist(prediction.float(), target.float(), compute_mode='donot_use_mm_for_euclid_dist').square()
    forward, fi = distance.min(dim=2)
    backward, bi = distance.min(dim=1)

    def term(distances, indices):
        counts = torch.zeros_like(distances)
        counts.scatter_add_(1, indices, torch.ones_like(distances))
        weight = counts.gather(1, indices).detach().pow(exponent).clamp_min(1).reciprocal()
        return (1 - torch.exp(-alpha * distances) * weight).mean(dim=1)

    return (term(forward, fi) + term(backward, bi)) / 2
