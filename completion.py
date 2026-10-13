"""轻量 PointNet 风格局部补全基线，不含文本条件。"""
import numpy as np
import torch
from torch import nn


class LocalCompletion(nn.Module):
    def __init__(self, output_points=128, output_mode='soft', decoder_width=256):
        super().__init__()
        self.output_points = output_points
        if output_mode not in ('soft', 'tanh'):
            raise ValueError('未知的球内输出方式。')
        self.output_mode = output_mode
        if decoder_width < 1:
            raise ValueError('解码器宽度必须为正。')
        self.encoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 128),
                                     nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(128 + 4, decoder_width), nn.ReLU(),
                                     nn.Linear(decoder_width, output_points * 3))

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
    config = checkpoint.get('config', {})
    model = make_model(checkpoint['output_points'], checkpoint.get('output_mode', 'soft'),
                       checkpoint.get('architecture', config.get('architecture', 'mlp')),
                       checkpoint.get('coarse_points', config.get('coarse_points', 32)),
                       checkpoint.get('decoder_width', config.get('decoder_width', 256)))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint


class HierarchicalCompletion(LocalCompletion):
    """借鉴PF-Net父点+残差子点，32→128的两级简化；非完整论文复现。

    Same point encoder; no adversarial/multiscale encoder. Coarse centers and
    parent features condition a shared child-offset head. Ball constraint is
    an engineering adaptation for the selected local region.
    """
    def __init__(self, output_points=128, output_mode='tanh', coarse_points=32):
        if coarse_points < 1 or output_points % coarse_points or coarse_points >= output_points:
            raise ValueError('分层输出需细点数为粗点数的整数倍，且细点数更多。')
        super().__init__(output_points, output_mode)
        del self.decoder
        self.coarse_points = coarse_points
        self.children_per_parent = output_points // coarse_points
        self.parent_decoder = nn.Sequential(nn.Linear(132, 256), nn.ReLU(),
                                            nn.Linear(256, coarse_points * 67))
        self.child_decoder = nn.Sequential(nn.Linear(64, 64), nn.ReLU(),
                                           nn.Linear(64, self.children_per_parent * 3))

    def forward_stages(self, points, region):
        center, radius = region[:, :3], region[:, 3:4]
        relative = (points-center[:, None, :]) / radius[:, None, :]
        feature = self.encoder(relative).amax(dim=1)
        parents = self.parent_decoder(torch.cat((feature, region), dim=1)).reshape(-1, self.coarse_points, 67)
        raw = parents[..., :3]
        norm = torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
        if self.output_mode == 'soft':
            coarse_local = raw / (1+norm)
        else:
            safe = norm.clamp_min(1e-6)
            coarse_local = raw * (torch.tanh(safe)/safe)
        offsets = self.child_decoder(parents[..., 3:]).reshape(-1, self.coarse_points, self.children_per_parent, 3)
        local = coarse_local[:, :, None, :] + .25 * torch.tanh(offsets)
        # 只裁剪球外子点，不对球内残差再次做tanh，避免额外径向收缩。
        local = local / torch.linalg.vector_norm(local, dim=-1, keepdim=True).clamp_min(1.)
        fine = center[:, None, :] + radius[:, None, :] * local.reshape(-1, self.output_points, 3)
        coarse = center[:, None, :] + radius[:, None, :] * coarse_local
        return coarse, fine

    def forward(self, points, region):
        return self.forward_stages(points, region)[1]


def make_model(output_points, output_mode, architecture='mlp', coarse_points=32, decoder_width=256):
    if architecture == 'mlp':
        return LocalCompletion(output_points, output_mode, decoder_width)
    if architecture == 'hierarchical':
        return HierarchicalCompletion(output_points, output_mode, coarse_points)
    raise ValueError('未知补全架构。')


@torch.no_grad()
def farthest_subset(points, count):
    """粗层训练目标的确定性FPS；只在训练时访问缺失真值。"""
    if points.ndim != 3 or points.shape[-1] != 3 or not 1 <= count <= points.shape[1]:
        raise ValueError('FPS目标点数无效。')
    batch, number, _ = points.shape
    distances = torch.full((batch, number), float('inf'), device=points.device)
    selected = torch.empty((batch, count), dtype=torch.long, device=points.device)
    current = torch.zeros(batch, dtype=torch.long, device=points.device)
    batch_index = torch.arange(batch, device=points.device)
    used = torch.zeros((batch, number), dtype=torch.bool, device=points.device)
    for index in range(count):
        selected[:, index] = current
        used[batch_index, current] = True
        distance = (points-points[batch_index, current][:, None]).square().sum(dim=-1)
        distances = torch.minimum(distances, distance)
        current = distances.masked_fill(used, -1).argmax(dim=1)
    return points.gather(1, selected[..., None].expand(-1, -1, 3))
