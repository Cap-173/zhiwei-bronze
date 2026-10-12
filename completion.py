"""轻量 PointNet 风格局部补全基线，不含文本条件。"""
import numpy as np
import torch
from torch import nn


class LocalCompletion(nn.Module):
    def __init__(self, output_points=128):
        super().__init__()
        self.output_points = output_points
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
        local = raw / (1 + torch.linalg.vector_norm(raw, dim=-1, keepdim=True))
        return center[:, None, :] + radius[:, None, :] * local


def chamfer_per_sample(prediction, target):
    distances = torch.cdist(prediction, target).square()
    return distances.amin(dim=2).mean(dim=1) + distances.amin(dim=1).mean(dim=1)


def chamfer_squared(prediction, target):
    return chamfer_per_sample(prediction, target).mean()


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
    model = LocalCompletion(checkpoint['output_points'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint
