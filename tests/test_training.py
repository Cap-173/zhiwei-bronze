from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from damage import write_cloud
from evaluate import main as evaluate_main
from prepare_dataset import build_dataset
from train import resolve_device, train
from training_data import nearest_squared, read_manifest


class TrainingTests(unittest.TestCase):
    def test_hierarchical_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            models=[]
            for i in range(3):
                path=root/f'{i}.ply'
                write_cloud(path,np.random.default_rng(i+30).normal(size=(180,3)))
                models.append((str(i),path))
            dataset=root/'dataset'
            build_dataset(models,dataset,samples=2,point_limit=180)
            config=dict(input_points=32,output_points=16,target_points=16,batch_size=2,
                        device='cpu',output_mode='tanh',architecture='hierarchical',
                        coarse_points=4,coarse_weight=.01)
            whole=train(dataset,root/'whole',epochs=2,**config)
            train(dataset,root/'resumed',epochs=1,**config)
            resumed=train(dataset,root/'resumed',epochs=2,resume=root/'resumed/last.pt',**config)
            self.assertEqual(whole['history'],resumed['history'])
            with self.assertRaisesRegex(ValueError,'参数'):
                train(dataset,root/'resumed',epochs=3,resume=root/'resumed/last.pt',
                      **{**config,'coarse_weight':.02})

    def test_dcd_training_resume_and_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            models = []
            for i in range(3):
                path = root / f'{i}.ply'
                write_cloud(path, np.random.default_rng(i+20).normal(size=(150, 3)))
                models.append((str(i), path))
            dataset = root/'dataset'
            build_dataset(models, dataset, samples=1, point_limit=150)
            config = dict(input_points=32, output_points=16, target_points=16,
                          batch_size=1, device='cpu', output_mode='tanh', loss_mode='dcd')
            train(dataset, root/'run', epochs=1, **config)
            report = train(dataset, root/'run', epochs=2, resume=root/'run/last.pt', **config)
            self.assertEqual(report['config']['loss_mode'], 'dcd')
            self.assertEqual(len(report['history']), 2)
            with self.assertRaisesRegex(ValueError, '参数'):
                train(dataset, root/'run', epochs=3, resume=root/'run/last.pt',
                      **config, dcd_alpha=100.)

    def test_chunked_distance(self):
        a, b = torch.randn(13, 3), torch.randn(17, 3)
        expected = torch.cdist(a[None], b[None]).square()[0].amin(dim=1)
        torch.testing.assert_close(nearest_squared(a, b, chunk=4), expected, atol=1e-5, rtol=1e-5)

    def test_unavailable_cuda_is_explicit(self):
        if not torch.cuda.is_available():
            with self.assertRaisesRegex(ValueError, 'CUDA'):
                resolve_device('cuda')

    def test_resume_and_held_out_evaluation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            models = []
            for i in range(3):
                path = root / f'model_{i}.ply'
                write_cloud(path, np.random.default_rng(i).normal(size=(250, 3)))
                models.append((str(i), path))
            dataset = root / 'dataset'
            build_dataset(models, dataset, samples=3, point_limit=200)
            config = dict(input_points=32, output_points=16, batch_size=2, target_points=32, device='cpu')
            whole = train(dataset, root / 'whole', epochs=3, **config)
            train(dataset, root / 'resumed', epochs=1, **config)
            # 模拟新增消融参数之前的 v2 检查点，验证默认基线可继续恢复。
            legacy = torch.load(root/'resumed'/'last.pt', weights_only=True)
            legacy['config'].pop('output_mode')
            legacy['config'].pop('boundary_weight')
            legacy.pop('output_mode')
            torch.save(legacy, root/'resumed'/'last.pt')
            resumed = train(dataset, root / 'resumed', epochs=3, resume=root / 'resumed' / 'last.pt', **config)
            self.assertEqual(whole['history'], resumed['history'])
            a = torch.load(root / 'whole' / 'last.pt', weights_only=True)
            b = torch.load(root / 'resumed' / 'last.pt', weights_only=True)
            for name in a['state_dict']:
                torch.testing.assert_close(a['state_dict'][name], b['state_dict'][name], rtol=0, atol=0)
            self.assertEqual(a['selection_split'], 'val')
            result = evaluate_main(['--checkpoint', str(root/'whole'/'best.pt'), '--dataset', str(dataset)])
            self.assertEqual(result, 0)
            with self.assertRaisesRegex(ValueError, '参数'):
                train(dataset, root / 'resumed', epochs=4, resume=root / 'resumed' / 'last.pt',
                      **{**config, 'batch_size': 1})
            manifest_path = dataset / 'manifest.json'
            text = manifest_path.read_text()
            manifest_path.write_text(text + ' ')
            # JSON 空白不改变数据语义，不影响哈希。
            self.assertEqual(read_manifest(dataset)['status'], 'complete')
            manifest = read_manifest(dataset)
            sample_path = dataset / manifest['samples'][0]['path'] / 'training.npz'
            with sample_path.open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, '数据集内容'):
                train(dataset, root / 'resumed', epochs=4, resume=root / 'resumed' / 'last.pt', **config)

    def test_cross_split_model_is_rejected(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'manifest.json').write_text(json.dumps(dict(status='complete', samples=[
                dict(model_id='same', split='train', path='a'),
                dict(model_id='same', split='test', path='b')])))
            with self.assertRaisesRegex(ValueError, '跨越'):
                read_manifest(root)
