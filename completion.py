"""轻量 PointNet 风格局部补全基线，不含文本条件。"""
import numpy as np
import torch
from torch import nn


class LocalCompletion(nn.Module):
    def __init__(self, output_points=128, output_mode='soft'):
        super().__init__()
        self.output_points = output_points
        if output_mode not in ('soft', 'tanh'):
            raise ValueError('未知的球内输出方式。')
        self.output_mode = output_mode
        self.encoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 128),
                                     nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(128 + 4, 256), nn.ReLU(),
                                     nn.Linear(256, output_points * 3))

    def forward(self, points, region):
        center, radius = region[:, :3], region[:, 3:4]
        relative = (points - center[:, None, :]) / radius[:, None, :]
        features = self.encoder(relative).amax(dim=1)
        raw = self.decoder(torch.cat((features, region), dim=1)).reshape(-1, self.output_points, 3)
        # 输出被限制在指定球内；保留区域由拼接原始点保证。
        norm = torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
        if self.output_mode == 'soft':
            local = raw / (1 + norm)
        else:
            # 相同原始幅度能到达更外侧，保持球内限制且不改变向量方向。
            safe_norm = norm.clamp_min(1e-6)
            local = raw * (torch.tanh(safe_norm) / safe_norm)
        return center[:, None, :] + radius[:, None, :] * local


def chamfer_per_sample(prediction, target):
    distances = torch.cdist(prediction, target).square()
    return distances.amin(dim=2).mean(dim=1) + distances.amin(dim=1).mean(dim=1)


def chamfer_squared(prediction, target):
    return chamfer_per_sample(prediction, target).mean()


def boundary_coverage_loss(prediction, target, region):
    """真实外缘点到预测点的单向平方距离均值；没有外缘点时为零。"""
    distances = torch.cdist(prediction, target).square().amin(dim=1)
    radial = torch.linalg.vector_norm(target-region[:, None, :3], dim=-1)
    outer = radial >= region[:, 3:4] * 0.8
    return (distances * outer).sum(dim=1) / outer.sum(dim=1).clamp_min(1)


def sample_points(points, count, rng):
    if not len(points):
        raise ValueError('不能采样空点云。')
    return points[rng.choice(len(points), count, replace=len(points) < count)]


def context_points(points, center, count, rng):
    """局部邻居 + 全局上下文，仅访问可见点。"""
    distances = np.linalg.norm(points - center, axis=1)
    near = points[np.argsort(distances)[:min(len(points), count)]]
    local_count = count // 2
    return np.concatenate((sample_points(near, local_count, rng),
                           sample_points(points, count-local_count, rng)))


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = LocalCompletion(checkpoint['output_points'], checkpoint.get('output_mode', 'soft'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint
