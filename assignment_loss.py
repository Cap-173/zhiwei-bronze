"""Exact equal-mass Euclidean matching inspired by PCN section4.3.

SciPy uses a modified Jonker-Volgenant solver, not PCN's CUDA approximation.
Assignment is discrete/detached; matched Euclidean costs carry piecewise gradients.
"""
import torch


def assignment_per_sample(prediction, target, region):
    from scipy.optimize import linear_sum_assignment
    if prediction.ndim!=3 or prediction.shape[-1]!=3 or target.shape!=prediction.shape or prediction.shape[1]<1:
        raise ValueError('EMD对照需要非空、等点数B×N×3点集。')
    if region.shape!=(prediction.shape[0],4) or not torch.isfinite(region).all() or not (region[:,3]>0).all():
        raise ValueError('匹配区域半径必须为正且有限。')
    p,t=prediction.float(),target.float()
    if not torch.isfinite(p).all() or not torch.isfinite(t).all():
        raise ValueError('匹配点集需为有限坐标。')
    with torch.no_grad():
        costs=torch.cdist(p,t,compute_mode='donot_use_mm_for_euclid_dist').cpu().numpy()
        columns=[]
        for cost in costs:
            rows,cols=linear_sum_assignment(cost)
            # Square assignment: sorted row indices cover every predicted slot.
            if len(rows)!=p.shape[1]:raise ValueError('匹配未覆盖所有点。')
            columns.append(torch.tensor(cols,device=p.device,dtype=torch.long))
        indices=torch.stack(columns)
    matched=t.gather(1,indices[:,:,None].expand(-1,-1,3))
    return torch.linalg.vector_norm(p-matched,dim=-1).mean(dim=1)/region[:,3]
