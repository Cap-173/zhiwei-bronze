from pathlib import Path
import tempfile
import unittest

import numpy as np

from damage import write_cloud
from prepare_dataset import build_dataset, fit_transform, load_training_sample, normalize, restore, split_models


class DatasetTests(unittest.TestCase):
    def test_inverse_and_degenerate(self):
        points = np.array([[10., 20, 30], [15, 22, 35], [9, 24, 33]])
        offset, scale = fit_transform(points)
        np.testing.assert_allclose(restore(normalize(points, offset, scale), offset, scale), points)
        self.assertLessEqual(np.linalg.norm(normalize(points, offset, scale), axis=1).max(), 1.000001)
        with self.assertRaises(ValueError):
            fit_transform(np.zeros((3, 3)))

    def test_model_split(self):
        splits = split_models(['a', 'b', 'c', 'd', 'e'], 42)
        self.assertEqual(set(splits.values()), {'train', 'val', 'test'})
        self.assertEqual(splits, split_models(['a', 'b', 'c', 'd', 'e'], 42))
        self.assertEqual(split_models(['a'], 42), {'a': 'train'})

    def test_batch_roundtrip_reproducibility_and_duplicate_detection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            points = np.random.default_rng(9).normal(size=(400, 3))
            source = root / 'source.ply'
            write_cloud(source, points)
            manifests = []
            for name in ('first', 'second'):
                manifest = build_dataset([('a', source)], root / name, samples=2, point_limit=200)
                self.assertEqual(manifest['status'], 'complete')
                self.assertFalse(manifest['evaluation_ready'])
                manifests.append(manifest)
            for left, right in zip(manifests[0]['samples'], manifests[1]['samples']):
                a = load_training_sample(root / 'first' / left['path'] / 'training.npz')
                b = load_training_sample(root / 'second' / right['path'] / 'training.npz')
                for key in a:
                    np.testing.assert_array_equal(a[key], b[key])
                np.testing.assert_array_equal(a['input'], a['target'][~a['mask']])
                np.testing.assert_array_equal(a['local_target'], a['target'][a['mask']])
                raw = restore(a['target'], a['offset'], a['scale'])
                self.assertTrue(all(np.min(np.linalg.norm(points - p, axis=1)) < 1e-10 for p in raw))
            with self.assertRaises(FileExistsError):
                build_dataset([('a', source)], root / 'first')
            with self.assertRaisesRegex(ValueError, '相同坐标'):
                build_dataset([('a', source), ('b', source)], root / 'duplicates')
            self.assertFalse((root / 'duplicates').exists())
