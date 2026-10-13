"""支持 CPU/CUDA、批量训练和按完整轮数恢复的局部补全基线。"""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

from completion import make_model, farthest_subset, chamfer_per_sample, boundary_coverage_loss
from density_loss import density_chamfer
from main import ROOT
from training_data import CompletionDataset, dataset_fingerprint, evaluate_entries, read_manifest


def resolve_device(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA 不可用：请在 GPU 机器安装对应的 CUDA 版 PyTorch，并检查驱动。')
    if name not in ('cpu', 'cuda'):
        raise ValueError('设备必须是 auto、cpu 或 cuda。')
    return torch.device(name)


def atomic_checkpoint(checkpoint, path):
    temporary = path.with_suffix('.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def train(dataset, output, epochs=50, input_points=512, output_points=128, seed=42, lr=0.001,
          batch_size=8, target_points=256, device='auto', workers=0, amp=False, resume=None,
          output_mode='soft', boundary_weight=0., loss_mode='cd', dcd_alpha=50., dcd_exponent=0.5,
          architecture='mlp', coarse_points=32, coarse_weight=0., decoder_width=256):
    dataset, output = Path(dataset), Path(output)
    if min(epochs, output_points, batch_size, target_points) < 1 or input_points < 2 or workers < 0 or seed < 0 or not math.isfinite(lr) or lr <= 0:
        raise ValueError('训练参数无效：点数、批大小、轮数和学习率必须为正。')
    device = resolve_device(device)
    if amp and device.type != 'cuda':
        raise ValueError('混合精度 --amp 需要 CUDA。')
    if output_mode not in ('soft', 'tanh') or not math.isfinite(boundary_weight) or boundary_weight < 0:
        raise ValueError('输出方式或外缘约束权重无效。')
    if loss_mode not in ('cd', 'dcd'):
        raise ValueError('未知训练损失。')
    if architecture not in ('mlp', 'hierarchical') or not math.isfinite(coarse_weight) or coarse_weight < 0:
        raise ValueError('架构或粗层权重无效。')
    if architecture == 'mlp' and coarse_weight:
        raise ValueError('MLP架构没有粗层输出。')
    if decoder_width < 1 or (architecture == 'hierarchical' and decoder_width != 256):
        raise ValueError('解码器宽度无效；分层结构使用固定父点头。')
    if architecture == 'hierarchical' and (coarse_points < 1 or output_points % coarse_points
            or coarse_points >= output_points or coarse_points > target_points or loss_mode != 'cd' or boundary_weight):
        raise ValueError('分层实验需要有效点数、CD损失及无附加边缘约束。')
    if loss_mode == 'dcd' and (target_points != output_points or boundary_weight != 0
            or not math.isfinite(dcd_alpha) or dcd_alpha <= 0
            or not math.isfinite(dcd_exponent) or not 0 <= dcd_exponent <= 1):
        raise ValueError('DCD需要等点数、无边缘附加项及有效参数。')
    manifest = read_manifest(dataset)
    entries = {split: [e for e in manifest['samples'] if e['split'] == split] for split in ('train', 'val')}
    if not entries['train']:
        raise ValueError('没有训练样本。')
    fingerprint = dataset_fingerprint(dataset, manifest)
    config = dict(input_points=input_points, output_points=output_points, seed=seed, lr=lr,
                  batch_size=batch_size, target_points=target_points, amp=amp,
                  output_mode=output_mode, boundary_weight=boundary_weight)
    if loss_mode == 'dcd':
        config.update(loss_mode=loss_mode, dcd_alpha=dcd_alpha, dcd_exponent=dcd_exponent,
                      dcd_coordinate_space='normalized visible-input coordinates; no radius rescaling')
    if architecture == 'hierarchical':
        config.update(architecture=architecture, coarse_points=coarse_points, coarse_weight=coarse_weight,
                      child_offset_bound=0.25, coarse_target='FPS of sampled training target')
    elif decoder_width != 256:
        config.update(decoder_width=decoder_width)
    restored = None
    if resume:
        resume = Path(resume).resolve()
        if resume.name != 'last.pt' or output.resolve() != resume.parent:
            raise ValueError('续训必须使用 last.pt，并写回同一个运行目录。')
        restored = torch.load(resume, map_location='cpu', weights_only=True)
        if restored.get('training_version') != 2:
            raise ValueError('旧版模型仅支持推理，不能完整续训；请开始新的训练。')
        if restored['dataset_fingerprint'] != fingerprint:
            raise ValueError('数据集内容变化，拒绝续训。')
        previous_config = dict(output_mode='soft', boundary_weight=0., **restored['config']) if 'output_mode' not in restored['config'] else restored['config']
        if previous_config != config:
            raise ValueError('续训参数与检查点不一致，请沿用原训练配置。')
        if epochs <= restored['epoch']:
            raise ValueError('--epochs 是总轮数，必须大于已完成轮数。')
        if not (output / 'best.pt').exists():
            raise ValueError('续训目录缺少 best.pt，请连同整个运行目录一起迁移。')
    else:
        output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    model = make_model(output_points, output_mode, architecture, coarse_points, decoder_width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    history, best, first_epoch = [], float('inf'), 1
    if restored:
        model.load_state_dict(restored['state_dict'])
        optimizer.load_state_dict(restored['optimizer'])
        scaler.load_state_dict(restored['scaler'])
        torch.set_rng_state(restored['torch_rng'])
        if device.type == 'cuda' and restored['cuda_rng']:
            torch.cuda.set_rng_state_all(restored['cuda_rng'])
        history, best, first_epoch = restored['history'], restored['best_score'], restored['epoch'] + 1

    def evaluate(split):
        return evaluate_entries(model, dataset, entries[split], input_points, seed+1, device)

    initial = restored['initial_train_cd_squared'] if restored else evaluate('train')['mean_local_cd_squared']
    data = CompletionDataset(dataset, entries['train'], input_points, target_points, seed)
    print(f'Device={device} | batch={batch_size} | AMP={amp} | start epoch={first_epoch}', flush=True)
    for epoch in range(first_epoch, epochs+1):
        data.epoch = epoch
        generator = torch.Generator().manual_seed(seed+epoch)
        loader = DataLoader(data, batch_size=batch_size, shuffle=True, generator=generator,
                            num_workers=workers, pin_memory=device.type == 'cuda')
        model.train()
        loss_sum, seen = 0., 0
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        for points, region, target in loader:
            points, region, target = (value.to(device, non_blocking=True) for value in (points, region, target))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                if architecture == 'hierarchical':
                    coarse, prediction = model.forward_stages(points, region)
                else:
                    prediction = model(points, region)
            # 距离与尺度运算保持 float32，降低半精度数值风险。
            if loss_mode == 'dcd':
                # DCD为有界指数损失，不再按R²缩放；alpha作用于输入归一化坐标。
                loss = density_chamfer(prediction, target, dcd_alpha, dcd_exponent).mean()
            else:
                per_sample = chamfer_per_sample(prediction.float(), target)
                if boundary_weight:
                    per_sample = per_sample + boundary_weight * boundary_coverage_loss(prediction.float(), target, region)
                if architecture == 'hierarchical' and coarse_weight:
                    coarse_target = farthest_subset(target, coarse_points)
                    per_sample = per_sample + coarse_weight * chamfer_per_sample(coarse.float(), coarse_target)
                loss = (per_sample / region[:, 3].square()).mean()
            if not torch.isfinite(loss):
                raise ValueError('训练损失非有限，请检查数据。')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(points)
            seen += len(points)
        training, validation = evaluate('train'), evaluate('val')
        train_cd = training['mean_local_cd_squared']
        val_cd = validation['mean_local_cd_squared'] if validation else None
        history.append(dict(epoch=epoch, train_loss=loss_sum/seen, train_cd_squared=train_cd,
                            val_cd_squared=val_cd, train_fscore=training['mean_local_fscore'],
                            val_fscore=validation['mean_local_fscore'] if validation else None,
                            cuda_peak_memory_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else None))
        score = val_cd if val_cd is not None else train_cd
        improved = score < best
        best = min(best, score)
        checkpoint = dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                          output_points=output_points, input_points=input_points, seed=seed, epoch=epoch,
                          dataset=str(dataset.resolve()), selection_split='val' if validation else 'train',
                          model_version=3 if architecture == 'hierarchical' else (1 if output_mode == 'soft' else 2), output_mode=output_mode,
                          training_version=2, config=config, dataset_fingerprint=fingerprint,
                          best_score=best, initial_train_cd_squared=initial, history=history,
                          torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [])
        # best 先写入，last 是完整轮次的恢复入口。
        if improved:
            atomic_checkpoint(checkpoint, output / 'best.pt')
        atomic_checkpoint(checkpoint, output / 'last.pt')
        report = dict(initial_train_cd_squared=initial, history=history, config=config,
                      device=str(device), evaluation_scope='held-out models' if validation else 'training samples only',
                      metric='bidirectional mean squared nearest distance in normalized coordinates; F-score threshold=0.02')
        (output / 'history.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        if epoch == first_epoch or epoch % 10 == 0 or epoch == epochs:
            print(f'Epoch {epoch}/{epochs} | loss={loss_sum/seen:.6f} | train CD2={train_cd:.6f}'
                  + (f' | val CD2={val_cd:.6f}' if val_cd is not None else ''), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--input-points', type=int, default=512)
    parser.add_argument('--output-points', type=int, default=128)
    parser.add_argument('--target-points', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--output-mode', choices=['soft', 'tanh'], default='soft')
    parser.add_argument('--boundary-weight', type=float, default=0.)
    parser.add_argument('--loss-mode', choices=['cd', 'dcd'], default='cd')
    parser.add_argument('--dcd-alpha', type=float, default=50.)
    parser.add_argument('--dcd-exponent', type=float, default=0.5)
    parser.add_argument('--architecture', choices=['mlp', 'hierarchical'], default='mlp')
    parser.add_argument('--coarse-points', type=int, default=32)
    parser.add_argument('--coarse-weight', type=float, default=0.)
    parser.add_argument('--decoder-width', type=int, default=256)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    try:
        restored = torch.load(args.resume, map_location='cpu', weights_only=True) if args.resume else None
        dataset = args.dataset or (Path(restored['dataset']) if restored else None)
        if dataset is None:
            datasets = sorted((ROOT / 'outputs' / 'datasets').glob('dataset_*'))
            if not datasets:
                raise ValueError('请先运行 prepare_dataset.py。')
            dataset = datasets[-1]
        output = args.output_dir or (args.resume.resolve().parent if args.resume else ROOT / 'outputs' / 'runs' / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f'))
        report = train(dataset, output, args.epochs, args.input_points, args.output_points, args.seed, args.lr,
                       args.batch_size, args.target_points, args.device, args.workers, args.amp, args.resume,
                       args.output_mode, args.boundary_weight, args.loss_mode, args.dcd_alpha, args.dcd_exponent,
                       args.architecture, args.coarse_points, args.coarse_weight, args.decoder_width)
        print(f'模型保存: {output.resolve()} | 评价范围: {report["evaluation_scope"]}')
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f'训练失败: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
